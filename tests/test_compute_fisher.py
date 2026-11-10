"""Tests for compute_fisher.py's own CLI wiring -- the piece unit tests on
continual.py's primitives can't see: that `--data-dir` actually reaches
load_dataset, and that Fisher is computed on the train-only split, not the
full dataset or the selection/held-out subsets.
"""

from __future__ import annotations

import sys

import torch

import compute_fisher
from continual import FisherPayload, checkpoint_fingerprint
from data import materialize_synthetic_shard
from model import ToyRegressor


def test_compute_fisher_cli_uses_data_dir_and_train_only_split(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    materialize_synthetic_shard(data_dir, num_samples=20, input_dim=16, seed=0)

    model = ToyRegressor()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = {"model_state": model.state_dict(), "step": 3}
    torch.save(checkpoint, checkpoint_path)

    seen_sample_counts = []
    real_compute_fisher_diag = compute_fisher.compute_fisher_diag

    def spy(model_arg, data_loader, loss_fn, num_batches=None):
        seen_sample_counts.append(sum(batch[0].size(0) for batch in data_loader))
        return real_compute_fisher_diag(model_arg, data_loader, loss_fn, num_batches=num_batches)

    monkeypatch.setattr(compute_fisher, "compute_fisher_diag", spy)

    output_path = tmp_path / "fisher.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_fisher.py",
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--data-dir",
            str(data_dir),
            "--batch-size",
            "4",
        ],
    )
    compute_fisher.main()

    # 20 synthetic samples, default 80/10/10 split -> 16 train / 2 selection / 2
    # held-out. If this regresses to the full dataset (20) or a selection/
    # held-out subset (2), this is the number that would catch it.
    assert seen_sample_counts == [16]

    payload = FisherPayload.load(output_path)
    assert payload.checkpoint_fingerprint == checkpoint_fingerprint(checkpoint["model_state"])
