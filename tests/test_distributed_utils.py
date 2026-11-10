"""CPU-only tests for distributed_utils, run with gloo and world_size=2."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import torch
import torch.multiprocessing as mp

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed_utils import (
    CheckpointHandoff,
    broadcast_exit_on_mismatch,
    find_free_port,
    install_graceful_shutdown_handler,
    setup_process_group,
    teardown_process_group,
)
from train import microbatched_step


def _run_worker(rank, world_size, master_port, target, result_dir, extra) -> None:
    setup_process_group(rank, world_size, backend="gloo", master_port=master_port)
    try:
        result = target(rank, world_size, extra)
        (Path(result_dir) / f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        teardown_process_group()


def _spawn_and_collect(target, world_size: int, result_dir, extra=None):
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    master_port = find_free_port()
    mp.spawn(
        _run_worker,
        args=(world_size, master_port, target, str(result_dir), extra),
        nprocs=world_size,
        join=True,
    )
    return [json.loads((Path(result_dir) / f"rank{r}.json").read_text()) for r in range(world_size)]


def _mismatch_worker(rank, world_size, extra):
    local_ok = rank != 0  # rank 0 reports unhealthy
    healthy = broadcast_exit_on_mismatch(local_ok, rank, device=torch.device("cpu"))
    return {"rank": rank, "healthy": healthy}


def test_broadcast_exit_on_mismatch_is_unanimous(tmp_path):
    results = _spawn_and_collect(_mismatch_worker, world_size=2, result_dir=tmp_path)
    assert all(r["healthy"] is False for r in results), results


def _healthy_worker(rank, world_size, extra):
    healthy = broadcast_exit_on_mismatch(True, rank, device=torch.device("cpu"))
    return {"rank": rank, "healthy": healthy}


def test_broadcast_exit_on_mismatch_all_healthy(tmp_path):
    results = _spawn_and_collect(_healthy_worker, world_size=2, result_dir=tmp_path)
    assert all(r["healthy"] is True for r in results), results


def _checkpoint_worker(rank, world_size, extra):
    handoff = CheckpointHandoff(Path(extra))
    if rank == 0:
        handoff.save({"value": 42}, step=1)
    return {"rank": rank}


def test_checkpoint_handoff_survives_crash_mid_write(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    _spawn_and_collect(_checkpoint_worker, world_size=2, result_dir=tmp_path / "results", extra=str(checkpoint_dir))

    # Simulate a crash mid-write: a stray .tmp file must never be picked up as "latest".
    stray_tmp = checkpoint_dir / "latest.pt.tmp"
    stray_tmp.write_bytes(b"not a valid checkpoint")

    handoff = CheckpointHandoff(checkpoint_dir)
    loaded = handoff.load_latest()
    assert loaded == {"value": 42}


def _lockstep_bad_batch_worker(rank, world_size, extra):
    """Mirrors train.py run_phase1's per-iteration pattern: check health (a
    collective) before the step that triggers DDP's own collectives, on
    every iteration. Only rank 0 ever sees bad data, and only at one
    iteration -- every rank must abort at that exact same iteration, having
    taken the exact same number of training steps beforehand.
    """
    model = nn.Linear(4, 1)
    ddp_model = DDP(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    num_iterations = 5
    bad_at_iteration = 2
    steps_taken = 0
    aborted_at = None
    for i in range(num_iterations):
        inputs = torch.randn(4, 4)
        targets = torch.randn(4, 1)
        if rank == 0 and i == bad_at_iteration:
            inputs[0, 0] = float("nan")

        local_ok = bool(torch.isfinite(inputs).all() and torch.isfinite(targets).all())
        healthy = broadcast_exit_on_mismatch(local_ok, rank, device=torch.device("cpu"))
        if not healthy:
            aborted_at = i
            break
        microbatched_step(ddp_model, optimizer, loss_fn, inputs, targets, microbatch_size=2)
        steps_taken += 1

    return {"rank": rank, "steps_taken": steps_taken, "aborted_at": aborted_at}


def test_lockstep_health_check_prevents_rank_divergence(tmp_path):
    """Regression test for a real deadlock bug: when the health check ran only
    once per epoch (after the batch loop), a rank that saw bad data could
    break out of its local loop before its peers learned about it, leaving
    the healthy ranks still issuing DDP collective calls the exited rank no
    longer matched -- a hang. Checking before every step keeps every rank in
    lockstep on the number of collective calls issued either way.
    """
    results = _spawn_and_collect(_lockstep_bad_batch_worker, world_size=2, result_dir=tmp_path)
    assert all(r["aborted_at"] == 2 for r in results), results
    assert all(r["steps_taken"] == 2 for r in results), results


def test_graceful_shutdown_handler_runs_once():
    calls = []
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)
    try:
        install_graceful_shutdown_handler(lambda: calls.append("shutdown"))
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)  # second signal must not re-run the hook
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)
    assert calls == ["shutdown"]
