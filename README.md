# DDP Training Harness

Small, dependency-light patterns for running long, unattended multi-GPU
training jobs on a shared cluster — built to survive the failure modes that
actually show up in practice, not just the happy path.

## Why this exists

On a shared SLURM-style cluster, a training job usually doesn't end because
training finished — it ends because the walltime budget ran out. I rely on
that constraint directly: instead of early stopping, I let jobs run to the
walltime limit and resume from a checkpoint in the next job, by choice. That
only works if the harness can (a) tell when something has gone quietly wrong
on one rank instead of hanging forever, and (b) handle `SIGTERM` (which SLURM
sends before the hard kill) by checkpointing and tearing down cleanly rather
than losing the run.

This repo is a from-scratch, sanitized demo of those patterns — a toy model,
synthetic data, and the harness code — not the training code for any
specific project.

## What's inside

| File | Pattern |
|---|---|
| `distributed_utils.py` | process-group setup/teardown via a **monitored barrier** (backed by a side-channel gloo group even when the main backend is nccl, since nccl's own collectives can hang indefinitely on a stuck rank, while gloo's `monitored_barrier` supports a timeout and names the unresponsive rank); per-rank timestamped hang-debug logging around collectives; `SIGTERM`/`SIGINT` handlers that run a shutdown hook exactly once; an all-reduce "health flag" so a dtype/shape/NaN problem seen by only one rank makes every rank exit together instead of deadlocking; a rank-0 checkpoint writer that's crash-safe (temp file + atomic rename) |
| `continual.py` | a diagonal Fisher-information estimate; a provenance-tagged Fisher payload (a later run refuses to load one computed from the wrong checkpoint); an EWC penalty anchored to a reference checkpoint; an SWA tracker that keeps both the running weight-average *and* the best single snapshot by validation metric, and ships whichever one actually wins rather than assuming the average is better by construction |
| `compute_fisher.py` | standalone, single-process script: load a checkpoint, verify it's fp32, compute the Fisher diagonal, save a provenance-tagged payload — not distributed, since it's a one-shot post-training pass, not a training job |
| `model.py` / `data.py` | a toy MLP regressor and a synthetic dataset — stand-ins for the real job this pattern set was built for |
| `train.py` | two phases. **Phase 1**: DDP training with a genuine three-way split — a *selection* set used only to pick the best checkpoint, and a *held-out* exam set that never influences any model-selection decision. **Phase 2**: resumes from a phase-1 checkpoint, adds the EWC penalty (using a Fisher payload from `compute_fisher.py`) and runs a long constant-LR SWA window. Spawns workers itself via `torch.multiprocessing.spawn` (`nccl` if GPUs are visible, `gloo` fallback on CPU) rather than an external launcher like `torchrun`; wires in gradient microbatching via `no_sync()` accumulation |

## Running it

```bash
pip install -r requirements.txt

# Phase 1: train from scratch (CPU, gloo, 2 simulated ranks)
python train.py --phase 1 --world-size 2 --epochs 5 --checkpoint-dir checkpoints

# Compute a Fisher payload from the phase-1 checkpoint
python compute_fisher.py --checkpoint checkpoints/latest.pt --output checkpoints/fisher.pt

# Phase 2: EWC + SWA refinement
python train.py --phase 2 --world-size 2 --swa-epochs 5 \
    --resume-from checkpoints/latest.pt --fisher-payload checkpoints/fisher.pt \
    --checkpoint-dir checkpoints

# Any of the above also runs on a GPU box by raising --world-size to num_gpus
```

## Tests / CI

```bash
pytest
```

Tests run CPU-only (`gloo`, `world_size=2`) so they don't need GPU hardware,
and run in GitHub Actions on every push.

## Status

Core patterns are implemented: process-group setup/teardown with a monitored
barrier, graceful `SIGTERM`/`SIGINT` shutdown, the health-mismatch guard,
crash-safe checkpointing, gradient microbatching, and the Fisher/EWC/SWA
pipeline across `train.py` + `compute_fisher.py`.

## License

MIT, see `LICENSE`.
