"""Tests for the Fisher/EWC/SWA logic in continual.py -- no distributed setup needed."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from continual import (
    ElasticWeightConsolidation,
    FisherPayload,
    SwaState,
    checkpoint_fingerprint,
    compute_fisher_diag,
)


def _tiny_model() -> nn.Module:
    return nn.Linear(4, 1)


def test_compute_fisher_diag_shapes_match_parameters():
    model = _tiny_model()
    inputs = torch.randn(32, 4)
    targets = torch.randn(32, 1)
    loader = [(inputs[i : i + 8], targets[i : i + 8]) for i in range(0, 32, 8)]
    fisher = compute_fisher_diag(model, loader, nn.MSELoss())
    for name, p in model.named_parameters():
        assert fisher[name].shape == p.shape
        assert fisher[name].dtype == p.dtype  # cast back from the float64 accumulator
        assert torch.all(fisher[name] >= 0)


def test_compute_fisher_diag_matches_mean_of_per_sample_squared_gradients():
    """Regression test: squaring a batch-mean gradient instead of averaging
    per-sample squared gradients is a different, systematically smaller
    quantity. Verify against a closed-form per-sample gradient with
    batch_size > 1, where the two diverge, and confirm the result is NOT the
    (wrong) batch-mean-then-square quantity either.
    """
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)

    xs = torch.tensor([[1.0], [2.0], [3.0], [-1.0]])
    ys = torch.tensor([[2.0], [1.0], [0.0], [1.0]])
    loader = [(xs, ys)]  # one batch holding all 4 samples
    fisher = compute_fisher_diag(model, loader, nn.MSELoss())

    w = model.weight.item()
    # MSELoss(reduction="mean") over a single sample with 1 output element is
    # (w*x - y)^2, so d(loss)/dw = 2*x*(w*x - y) per sample.
    per_sample_grads = [2 * x.item() * (w * x.item() - y.item()) for x, y in zip(xs, ys)]
    expected = sum(g**2 for g in per_sample_grads) / len(per_sample_grads)
    assert fisher["weight"].item() == pytest.approx(expected, rel=1e-5)

    wrong_quantity = (sum(per_sample_grads) / len(per_sample_grads)) ** 2
    assert fisher["weight"].item() != pytest.approx(wrong_quantity, rel=1e-5)


def test_fisher_payload_provenance_rejects_mismatched_checkpoint():
    model = _tiny_model()
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters()}
    fingerprint = checkpoint_fingerprint(model.state_dict())
    payload = FisherPayload(fisher_diag=fisher, checkpoint_fingerprint=fingerprint, source_step=1)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fisher.pt"
        payload.save(path)

        loaded = FisherPayload.load_matching(path, expected_fingerprint=fingerprint)
        assert loaded.checkpoint_fingerprint == fingerprint

        other_model = _tiny_model()
        other_fingerprint = checkpoint_fingerprint(other_model.state_dict())
        with pytest.raises(ValueError):
            FisherPayload.load_matching(path, expected_fingerprint=other_fingerprint)


def test_ewc_penalty_is_zero_at_reference_point():
    model = _tiny_model()
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters()}
    ewc = ElasticWeightConsolidation(model, fisher, importance=1.0)
    assert ewc.penalty().item() == 0.0

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    assert ewc.penalty().item() > 0.0


def test_ewc_rejects_missing_fisher_parameters():
    model = _tiny_model()
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters()}
    del fisher["bias"]
    with pytest.raises(ValueError, match="missing parameters"):
        ElasticWeightConsolidation(model, fisher, importance=1.0)


def test_ewc_rejects_extra_fisher_parameters():
    model = _tiny_model()
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters()}
    fisher["not_a_real_param"] = torch.ones(1)
    with pytest.raises(ValueError, match="aren't trainable parameters"):
        ElasticWeightConsolidation(model, fisher, importance=1.0)


def test_ewc_clips_outlier_and_mean_normalizes_fisher():
    model = _tiny_model()
    weight_fisher = torch.tensor([[1.0, 1.0, 1.0, 1000.0]])
    bias_fisher = torch.tensor([1.0])
    fisher = {"weight": weight_fisher, "bias": bias_fisher}

    raw_flat = torch.cat([weight_fisher.reshape(-1), bias_fisher.reshape(-1)])
    unclipped_normalized_outlier = (1000.0 / raw_flat.mean()).item()

    ewc = ElasticWeightConsolidation(model, fisher, importance=1.0, clip_percentile=80.0)

    processed_flat = torch.cat([t.reshape(-1) for t in ewc.fisher_diag.values()])
    assert torch.isclose(processed_flat.mean(), torch.tensor(1.0), atol=1e-5)

    outlier_value = ewc.fisher_diag["weight"][0, 3].item()
    assert outlier_value < unclipped_normalized_outlier  # clipping actually reduced it

    small_values = torch.cat([ewc.fisher_diag["weight"][0, :3], ewc.fisher_diag["bias"]])
    assert torch.allclose(small_values, small_values[0].expand_as(small_values))


def test_ewc_rejects_invalid_clip_percentile():
    model = _tiny_model()
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="clip_percentile"):
        ElasticWeightConsolidation(model, fisher, clip_percentile=0.0)
    with pytest.raises(ValueError, match="clip_percentile"):
        ElasticWeightConsolidation(model, fisher, clip_percentile=150.0)


def test_ewc_rejects_all_zero_fisher():
    model = _tiny_model()
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    with pytest.raises(RuntimeError, match="Post-clip Fisher mean"):
        ElasticWeightConsolidation(model, fisher)


def test_swa_state_ships_whichever_candidate_actually_wins():
    model = _tiny_model()
    swa = SwaState(model=model)

    def fake_evaluate(state_dict: dict) -> float:
        value = next(iter(state_dict.values())).flatten()[0].item()
        return abs(value - 2.0)  # weights near 2.0 score low (better)

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    swa.update_average()
    swa.consider_best(selection_metric=fake_evaluate(model.state_dict()))  # scores 1.0

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)
    swa.update_average()
    improved = swa.consider_best(selection_metric=fake_evaluate(model.state_dict()))  # scores 0.0
    assert improved

    # average of [1.0, 2.0] = 1.5 -> score 0.5; best snapshot (2.0) -> score 0.0 -> best wins
    shipped = swa.pick_shipping_state(fake_evaluate)
    shipped_value = next(iter(shipped.values())).flatten()[0].item()
    assert shipped_value == 2.0


def test_swa_state_seeded_with_prior_best_wont_ship_a_regression():
    """run_phase2 seeds SwaState's best with the resumed checkpoint's own
    recorded loss -- otherwise, if neither the SWA average nor anything
    seen during phase 2 beats that starting point, pick_shipping_state
    would still ship one of them anyway: a regression against what phase 1
    already achieved. This mirrors that seeding directly on SwaState.
    """
    model = _tiny_model()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    phase1_state = {k: v.clone() for k, v in model.state_dict().items()}

    swa = SwaState(model=model)
    swa.consider_best(selection_metric=0.1)  # the phase-1 checkpoint's own recorded loss

    with torch.no_grad():
        for p in model.parameters():
            p.fill_(99.0)  # phase 2 made things much worse
    swa.update_average()

    shipped = swa.pick_shipping_state(evaluate_fn=lambda _state: 50.0)  # worse than 0.1 either way

    for name, value in shipped.items():
        assert torch.equal(value, phase1_state[name]), name
