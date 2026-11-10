"""Tests for train.py's CLI wiring -- phase-1 resume-from-checkpoint.

Runs the real CLI end-to-end (via mp.spawn, world_size=1, CPU/gloo), not a
mock -- the thing worth regression-testing here is a structural guarantee
(epoch numbering can never go backward after a resume), which holds
regardless of the actual training dynamics on any given run.
"""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import compute_fisher
import train
from distributed_utils import find_free_port, setup_process_group, teardown_process_group
from model import ToyRegressor


def test_phase1_resume_continues_epoch_numbering_without_overwriting_archive(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "ckpt"
    argv_base = [
        "train.py",
        "--phase", "1",
        "--world-size", "1",
        "--epochs", "1",
        "--checkpoint-dir", str(checkpoint_dir),
    ]

    monkeypatch.setattr(sys, "argv", argv_base)
    train.main()

    step0_path = checkpoint_dir / "checkpoint_step00000000.pt"
    # Epoch 0 always "improves" over best_selection_loss=inf, so this file's
    # existence after a first run is guaranteed, not a training-dynamics fluke.
    assert step0_path.exists()
    step0_bytes_before = step0_path.read_bytes()

    monkeypatch.setattr(sys, "argv", argv_base + ["--resume-from", str(checkpoint_dir / "latest.pt")])
    train.main()

    # Resuming must continue epoch numbering from the checkpoint's own step
    # (not restart at 0) -- otherwise this second run's own epoch 0 would
    # silently overwrite the first run's checkpoint_step00000000.pt with
    # unrelated content from a different point in training. This holds no
    # matter what the resumed run's own loss does, since the fix makes epoch
    # 0 structurally unreachable on a resumed run, not just unlikely to recur.
    assert step0_path.read_bytes() == step0_bytes_before


def test_phase1_runs_with_world_size_2(tmp_path, monkeypatch):
    """CI's own world_size=2 coverage (via stage_and_run.sh) only asserts
    the process exits 0 -- a bash-level smoke test, not a pytest assertion
    on train.py's own multi-rank DDP path. This runs that same path
    directly inside pytest and checks it actually produced a checkpoint.
    """
    checkpoint_dir = tmp_path / "ckpt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py", "--phase", "1", "--world-size", "2", "--epochs", "1",
            "--checkpoint-dir", str(checkpoint_dir),
        ],
    )
    train.main()
    assert (checkpoint_dir / "latest.pt").exists()


def test_microbatched_step_matches_full_batch_gradient():
    """Accumulating gradients over microbatches under no_sync(), syncing
    only on the last one, must produce the same gradient as a single
    full-batch backward pass on the same data and starting weights --
    this is the correctness property the microbatching pattern depends
    on, independent of DDP's own cross-rank averaging (world_size=1 here,
    so that part is a no-op and isn't what's being tested).
    """
    port = find_free_port()
    setup_process_group(rank=0, world_size=1, backend="gloo", master_port=port)
    try:
        torch.manual_seed(0)
        microbatched_model = ToyRegressor()
        starting_state = {k: v.clone() for k, v in microbatched_model.state_dict().items()}
        inputs = torch.randn(10, 16)
        targets = torch.randn(10, 1)
        loss_fn = nn.MSELoss()

        ddp_model = DDP(microbatched_model)
        # lr=0.0: isolate the accumulated gradient itself, not the weights
        # a step would produce from it.
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.0)
        train.microbatched_step(ddp_model, optimizer, loss_fn, inputs, targets, microbatch_size=3)
        microbatched_grads = {n: p.grad.clone() for n, p in microbatched_model.named_parameters()}

        full_batch_model = ToyRegressor()
        full_batch_model.load_state_dict(starting_state)
        loss_fn(full_batch_model(inputs), targets).backward()

        for name, full_batch_param in full_batch_model.named_parameters():
            assert torch.allclose(microbatched_grads[name], full_batch_param.grad, atol=1e-6), name
    finally:
        teardown_process_group()


def test_phase2_requires_resume_from(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--phase", "2", "--world-size", "1"])
    with pytest.raises(SystemExit):
        train.main()


def test_phase2_ships_a_loadable_swa_checkpoint(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "ckpt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--phase", "1", "--world-size", "1", "--epochs", "1", "--checkpoint-dir", str(checkpoint_dir)],
    )
    train.main()

    fisher_path = checkpoint_dir / "fisher.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["compute_fisher.py", "--checkpoint", str(checkpoint_dir / "latest.pt"), "--output", str(fisher_path)],
    )
    compute_fisher.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--phase", "2",
            "--world-size", "1",
            "--swa-epochs", "1",
            "--resume-from", str(checkpoint_dir / "latest.pt"),
            "--fisher-payload", str(fisher_path),
            "--checkpoint-dir", str(checkpoint_dir),
        ],
    )
    train.main()

    # CI's own phase-2 smoke test only checks the process exits 0 -- it
    # doesn't check what, if anything, actually got shipped. This does:
    # the SWA checkpoint must exist and load into the real model class.
    shipped = torch.load(checkpoint_dir / "swa" / "latest.pt", map_location="cpu", weights_only=True)
    ToyRegressor().load_state_dict(shipped["model_state"])
