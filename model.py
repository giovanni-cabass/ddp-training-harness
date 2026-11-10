"""Toy model for the DDP training harness demo.

Deliberately simple and structurally unrelated to any real production
architecture -- its only job is to have learnable parameters and a loss,
so the distributed-training machinery around it (process groups,
checkpointing, fault handling) has something to exercise.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ToyRegressor(nn.Module):
    """A tiny 3-layer MLP regressor."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
