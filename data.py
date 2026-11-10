"""Synthetic dataset for the DDP training harness demo.

No real data of any kind -- just enough structure (a fixed linear ground
truth plus noise) that training loss is meaningful to look at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, TensorDataset


class SyntheticRegressionDataset(Dataset):
    def __init__(self, num_samples: int = 10_000, input_dim: int = 16, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(num_samples, input_dim, generator=generator)
        true_weights = torch.randn(input_dim, 1, generator=generator)
        noise = 0.1 * torch.randn(num_samples, 1, generator=generator)
        self.targets = self.inputs @ true_weights + noise

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


def materialize_synthetic_shard(output_dir: Path, num_samples: int = 10_000, input_dim: int = 16, seed: int = 0) -> None:
    """Write a synthetic dataset to disk as a toy "exported shard".

    Stands in for a real prepared-data export that would already be sitting
    on shared storage, ready to be staged to node-local scratch before
    training -- see stage_and_run.sh.
    """
    dataset = SyntheticRegressionDataset(num_samples=num_samples, input_dim=input_dim, seed=seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"inputs": dataset.inputs, "targets": dataset.targets}, output_dir / "shard_00.pt")


def load_dataset(data_dir: Optional[Path] = None, seed: int = 0) -> Dataset:
    """Load a staged shard from `data_dir` if given, else generate synthetic data in-memory."""
    if data_dir is None:
        return SyntheticRegressionDataset(seed=seed)
    payload = torch.load(Path(data_dir) / "shard_00.pt", map_location="cpu", weights_only=True)
    return TensorDataset(payload["inputs"], payload["targets"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Materialize a synthetic data shard on disk")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    materialize_synthetic_shard(args.output_dir, num_samples=args.num_samples, input_dim=args.input_dim, seed=args.seed)
    print(f"Wrote synthetic shard to {args.output_dir}")
