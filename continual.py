"""Continual-learning refinements layered on top of the base training loop.

Fisher information (diagonal approximation), EWC (Elastic Weight
Consolidation), and SWA (Stochastic Weight Averaging) -- all model-agnostic,
all bonus content on top of the core DDP harness.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fisher information (diagonal approximation)
# ---------------------------------------------------------------------------


def compute_fisher_diag(
    model: nn.Module,
    data_loader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Diagonal Fisher-information estimate: mean squared gradient per parameter.

    Single-process, read-only with respect to the model's weights (grads are
    cleared when done). `num_batches=None` consumes the whole loader once.
    """
    model.eval()
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
    seen = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        model.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
        seen += 1
    model.zero_grad(set_to_none=True)
    if seen == 0:
        raise ValueError("compute_fisher_diag: data_loader produced no batches")
    return {name: tensor / seen for name, tensor in fisher.items()}


def checkpoint_fingerprint(state_dict: Dict[str, torch.Tensor]) -> str:
    """Cheap, order-stable fingerprint of a state dict, for the Fisher provenance check."""
    hasher = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


@dataclass
class FisherPayload:
    """A Fisher-diagonal estimate, tagged with the checkpoint it came from.

    `checkpoint_fingerprint` is what later lets a training run refuse to
    load a Fisher payload computed from the wrong checkpoint.
    """

    fisher_diag: Dict[str, torch.Tensor]
    checkpoint_fingerprint: str
    source_step: int

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "fisher_diag": self.fisher_diag,
                "checkpoint_fingerprint": self.checkpoint_fingerprint,
                "source_step": self.source_step,
            },
            tmp_path,
        )
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "FisherPayload":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(**payload)

    @classmethod
    def load_matching(cls, path: Path, expected_fingerprint: str) -> "FisherPayload":
        """Load, but raise if the payload wasn't computed from this checkpoint."""
        payload = cls.load(path)
        if payload.checkpoint_fingerprint != expected_fingerprint:
            raise ValueError(
                "Fisher payload fingerprint mismatch: "
                f"expected {expected_fingerprint!r}, got {payload.checkpoint_fingerprint!r}. "
                "This Fisher estimate was computed from a different checkpoint."
            )
        return payload


# ---------------------------------------------------------------------------
# EWC penalty
# ---------------------------------------------------------------------------


class ElasticWeightConsolidation:
    """Penalizes drift away from a reference checkpoint, weighted by Fisher.

    loss = task_loss + (importance / 2) * sum_i F_i * (w_i - w_i_ref)^2
    """

    def __init__(self, model: nn.Module, fisher_diag: Dict[str, torch.Tensor], importance: float = 1.0) -> None:
        self.model = model
        self.fisher_diag = fisher_diag
        self.importance = importance
        self.reference_params = {
            name: p.detach().clone() for name, p in model.named_parameters() if name in fisher_diag
        }

    def penalty(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        total = torch.zeros((), device=device)
        for name, p in self.model.named_parameters():
            if name not in self.fisher_diag:
                continue
            fisher = self.fisher_diag[name].to(device)
            reference = self.reference_params[name].to(device)
            total = total + (fisher * (p - reference) ** 2).sum()
        return (self.importance / 2.0) * total


# ---------------------------------------------------------------------------
# SWA: constant-LR averaging window, tracked alongside the best single snapshot
# ---------------------------------------------------------------------------


@dataclass
class SwaState:
    """Tracks both the running weight-average and the best single snapshot.

    The two don't always agree on which to ship: compare them on the
    selection metric and keep whichever actually wins, rather than assuming
    the average is better by construction.
    """

    model: nn.Module
    average_state: Optional[Dict[str, torch.Tensor]] = None
    num_averaged: int = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric: float = float("inf")

    def update_average(self) -> None:
        current = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        if self.average_state is None:
            self.average_state = current
            self.num_averaged = 1
            return
        self.num_averaged += 1
        for key, value in current.items():
            self.average_state[key] = self.average_state[key] + (value - self.average_state[key]) / self.num_averaged

    def consider_best(self, selection_metric: float) -> bool:
        """Record the current weights as `best` if selection_metric improves on it."""
        if selection_metric < self.best_metric:
            self.best_metric = selection_metric
            self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            return True
        return False

    def pick_shipping_state(
        self, evaluate_fn: Callable[[Dict[str, torch.Tensor]], float]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the average against the recorded best (lower is better); return the winner."""
        if self.average_state is None:
            if self.best_state is None:
                raise ValueError("SwaState.pick_shipping_state: no snapshots recorded yet")
            return self.best_state
        if self.best_state is None:
            return self.average_state
        average_metric = evaluate_fn(self.average_state)
        return self.best_state if self.best_metric <= average_metric else self.average_state
