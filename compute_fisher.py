"""Standalone, single-process script: compute a diagonal Fisher-information
estimate at a saved checkpoint and save it as a provenance-tagged payload.

Deliberately not distributed -- this runs once, after phase-1 training, over
a single checkpoint, so there's no need for DDP machinery here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from continual import FisherPayload, checkpoint_fingerprint, compute_fisher_diag
from data import SyntheticRegressionDataset
from model import ToyRegressor


def verify_model_dtype_fp32(model: torch.nn.Module) -> None:
    bad = [name for name, p in model.named_parameters() if p.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"Expected fp32 parameters for Fisher computation, found non-fp32: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a diagonal Fisher-information payload from a checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ToyRegressor()
    model.load_state_dict(checkpoint["model_state"])
    verify_model_dtype_fp32(model)

    dataset = SyntheticRegressionDataset(seed=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_fn = torch.nn.MSELoss()

    fisher_diag = compute_fisher_diag(model, loader, loss_fn, num_batches=args.num_batches)
    payload = FisherPayload(
        fisher_diag=fisher_diag,
        checkpoint_fingerprint=checkpoint_fingerprint(checkpoint["model_state"]),
        source_step=checkpoint.get("step", -1),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload.save(args.output)
    print(f"Saved Fisher payload for checkpoint {args.checkpoint} -> {args.output}")


if __name__ == "__main__":
    main()
