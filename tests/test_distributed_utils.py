"""CPU-only tests for distributed_utils, run with gloo and world_size=2."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import torch
import torch.multiprocessing as mp

from distributed_utils import (
    CheckpointHandoff,
    find_free_port,
    install_graceful_shutdown_handler,
    setup_process_group,
    teardown_process_group,
)


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

    loaded = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=True)
    assert loaded == {"value": 42}


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
