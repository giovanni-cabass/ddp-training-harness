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
    """Diagonal Fisher-information estimate: mean of *per-sample* squared gradients.

    Single-process, read-only with respect to the model's weights (grads are
    cleared when done). `num_batches=None` consumes the whole loader once.

    Gradients are accumulated one sample at a time, not as a batch mean --
    squaring a batch-mean gradient is a different, systematically smaller
    quantity than the mean of per-sample squared gradients, and the gap is
    worst exactly where Fisher/EWC matters most: near a converged checkpoint,
    where individual-sample gradients are non-zero but their mean is close
    to zero. The running sum itself is kept in float64 and cast back to each
    parameter's own dtype only at the end (see below).

    Two mechanical differences from the real pipeline this generalizes, not
    just a reimplementation of it: it saves its Fisher payload in float64
    and leaves the downcast to the consumer at load time, whereas this
    downcasts before ever saving; and it takes a deterministic row-stride
    subsampling parameter and asserts it visited every expected training
    row exactly once, whereas `num_batches` here is a plain cutoff with no
    equivalent coverage check.
    """
    model.eval()
    # Accumulate in float64, not each parameter's own (fp32) dtype: summing
    # many per-sample squared-gradient terms means every addition rounds the
    # running total to fp32's ~7 decimal digits, and that rounding error
    # compounds over hundreds/thousands of summands -- float64 has enough
    # extra headroom that that compounding stays negligible across the whole
    # sum. Cast back to the parameter's own dtype at the end, so the result
    # matches the live model weights the EWC penalty is actually applied to.
    fisher = {
        name: torch.zeros_like(p, dtype=torch.float64) for name, p in model.named_parameters() if p.requires_grad
    }
    seen = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        for i in range(inputs.size(0)):
            model.zero_grad(set_to_none=True)
            prediction = model(inputs[i : i + 1])
            loss = loss_fn(prediction, targets[i : i + 1])
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach().double() ** 2
            seen += 1
    model.zero_grad(set_to_none=True)
    if seen == 0:
        raise ValueError("compute_fisher_diag: data_loader produced no batches")
    param_dtypes = {name: p.dtype for name, p in model.named_parameters()}
    return {name: (tensor / seen).to(param_dtypes[name]) for name, tensor in fisher.items()}


def checkpoint_fingerprint(state_dict: Dict[str, torch.Tensor]) -> str:
    """Cheap, order-stable fingerprint of a state dict, for the Fisher provenance check.

    A different mechanism than the real pipeline this generalizes, not just
    a reimplementation of it: the real provenance check matches on
    descriptive metadata (a checkpoint filename and a few bookkeeping
    fields), not a hash of the weights themselves. This hashes the actual
    tensor bytes, so it also catches a checkpoint silently overwritten with
    different weights under the same filename -- a stronger check, but a
    different one.
    """
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
        try:
            torch.save(
                {
                    "fisher_diag": self.fisher_diag,
                    "checkpoint_fingerprint": self.checkpoint_fingerprint,
                    "source_step": self.source_step,
                },
                tmp_path,
            )
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

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

    def __init__(
        self,
        model: nn.Module,
        fisher_diag: Dict[str, torch.Tensor],
        importance: float = 1.0,
        clip_percentile: float = 99.0,
    ) -> None:
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        missing = trainable_names - fisher_diag.keys()
        if missing:
            raise ValueError(
                f"ElasticWeightConsolidation: fisher_diag is missing parameters {sorted(missing)} -- "
                "this Fisher payload doesn't match the model; refusing to silently skip the EWC "
                "penalty for those parameters."
            )
        extra = fisher_diag.keys() - trainable_names
        if extra:
            raise ValueError(
                f"ElasticWeightConsolidation: fisher_diag has entries for {sorted(extra)}, which aren't "
                "trainable parameters of this model -- this Fisher payload doesn't match the model it's "
                "being applied to."
            )
        if not (0.0 < clip_percentile <= 100.0):
            raise ValueError(f"clip_percentile must be in (0, 100], got {clip_percentile}")

        # Clip the top percentile across the whole Fisher map, then
        # mean-normalize: without this, a handful of parameters with huge
        # Fisher values dominate the penalty and wash out everything else,
        # and the raw Fisher scale drifts from run to run -- normalizing
        # after clipping keeps `importance` meaning roughly the same
        # magnitude of penalty regardless of checkpoint/run.
        flat = torch.cat([t.reshape(-1) for t in fisher_diag.values()])
        threshold = torch.quantile(flat, clip_percentile / 100.0)
        clipped = {name: torch.clamp(t, max=threshold) for name, t in fisher_diag.items()}
        flat_clipped = torch.cat([t.reshape(-1) for t in clipped.values()])
        mean_clipped = flat_clipped.mean()
        if not torch.isfinite(mean_clipped) or mean_clipped <= 0:
            raise RuntimeError(
                f"Post-clip Fisher mean must be finite and > 0 for normalization, got {mean_clipped.item()}"
            )
        self.fisher_diag = {name: t / mean_clipped for name, t in clipped.items()}

        self.model = model
        self.importance = importance
        self.reference_params = {
            name: p.detach().clone() for name, p in model.named_parameters() if name in self.fisher_diag
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

    Averages the whole state_dict (parameters and buffers alike) with plain
    running-mean arithmetic, instead of using PyTorch's own
    `torch.optim.swa_utils.AveragedModel` the way the real pipeline this
    generalizes does. `AveragedModel` averages only `parameters()` and
    copies buffers through unchanged; this class's hand-rolled version
    doesn't make that distinction, so a non-floating-point buffer (e.g. a
    BatchNorm running-stats counter) would error or silently average
    incorrectly. `ToyRegressor` has no buffers, so this never triggers
    here -- but it's a gap specific to reimplementing the averaging by
    hand, not one the real pipeline has.
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
        """Evaluate the average against the recorded best (lower is better); return the winner.

        Automates a comparison that was a separate offline analysis step,
        done after training finished, in the real pipeline this
        generalizes -- not something the training script itself decided.
        """
        if self.average_state is None:
            if self.best_state is None:
                raise ValueError("SwaState.pick_shipping_state: no snapshots recorded yet")
            return self.best_state
        if self.best_state is None:
            return self.average_state
        average_metric = evaluate_fn(self.average_state)
        return self.best_state if self.best_metric <= average_metric else self.average_state
