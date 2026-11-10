# DDP Training Harness

Small, dependency-light patterns for running long, unattended multi-GPU
training jobs on a shared cluster — built to survive the failure modes that
actually show up in practice, not just the happy path.

## Why this exists

On a shared SLURM-style cluster, a training job usually doesn't end because
training finished — it ends because the walltime budget ran out. I lean into
that constraint rather than fighting it: instead of early-stopping on a
patience window, I estimate epochs-per-hour for a given architecture/dataset
and budget walltime accordingly, letting the job run until the scheduler ends
it and resuming from a checkpoint in the next one. Early stopping has a
sharper failure mode too, in my experience: a loss curve can plateau, look
converged, and then undergo a genuine transition to a lower plateau later on
— a patience window tuned for the first plateau stops before ever seeing
the second.

That only works if the harness can (a) tell when something has gone quietly
wrong on one rank instead of hanging forever, and (b) handle the walltime
kill itself without losing the run. Checkpointing every time the selection
metric improves covers most of that; the `SIGTERM` handler SLURM sends
before the hard kill adds one more checkpoint-and-clean-teardown attempt, as
a safety net for the rarer case where the kill lands between improvement
checks.

The same shared-cluster reality shapes how data gets to the job, too:
`stage_and_run.sh` copies input data to node-local scratch before training
starts and cleans it up on any exit, so a multi-epoch job doesn't hammer a
shared filesystem with repeated random-access reads, and nothing lingers on
a compute node other people also use.

This repo is a from-scratch, sanitized demo of those patterns — a toy model,
synthetic data, and the harness code — not the training code for any
specific project.

## What's inside

| File | Pattern |
|---|---|
| `distributed_utils.py` | Process-group robustness, in one module:<br>- **monitored barrier**, backed by a side-channel gloo group even when the main backend is nccl (nccl's own collectives can hang indefinitely on a stuck rank; gloo's `monitored_barrier` supports a timeout and names the unresponsive rank)<br>- per-rank timestamped hang-debug logging around collectives<br>- `SIGTERM`/`SIGINT` handlers that run a shutdown hook exactly once<br>- an all-reduce "health flag" so a dtype/shape/NaN problem seen by only one rank makes every rank exit together instead of deadlocking<br>- a rank-0 checkpoint writer that's crash-safe (temp file + atomic rename) |
| `continual.py` | a diagonal Fisher-information estimate; a provenance-tagged Fisher payload (a later run refuses to load one computed from the wrong checkpoint); an EWC penalty anchored to a reference checkpoint; an SWA tracker that keeps both the running weight-average *and* the best single snapshot by selection metric, and ships whichever one actually wins rather than assuming the average is better by construction |
| `compute_fisher.py` | standalone, single-process script: load a checkpoint, verify it's fp32, compute the Fisher diagonal on the train-only split, save a provenance-tagged payload — not distributed, since it's a one-shot post-training pass, not a training job |
| `model.py` / `data.py` | a toy MLP regressor and a synthetic dataset — stand-ins for the real job this pattern set was built for |
| `stage_and_run.sh` | node-local scratch staging: pre-flight disk-space check, `rsync` from shared storage to scratch, `trap ... EXIT`-based cleanup that runs on success/failure/`SIGTERM` alike, and a safety guard refusing to operate on anything that doesn't look like disposable scratch |
| `train.py` | two phases, spawned via `torch.multiprocessing.spawn` (`nccl` if GPUs are visible, `gloo` fallback on CPU) rather than an external launcher like `torchrun`:<br>- **Phase 1**: DDP training with a genuine three-way split — a *selection* set used only to pick the best checkpoint, and a *held-out* exam set that never influences any model-selection decision<br>- **Phase 2**: resumes from a phase-1 checkpoint, adds the EWC penalty (using a Fisher payload from `compute_fisher.py`) and runs a long constant-LR SWA window<br>- gradient microbatching via `no_sync()` accumulation, to simulate a larger effective batch size under a memory ceiling — DDP gradients sync only on the final microbatch of each step |

## Running it

```bash
pip install -r requirements.txt

# Phase 1: train from scratch (CPU, gloo, 2 simulated ranks)
python train.py --phase 1 --world-size 2 --epochs 5 --checkpoint-dir checkpoints

# Same thing, via the node-local staging wrapper (what actually runs on a
# cluster -- stages data to scratch first, cleans up on any exit)
./stage_and_run.sh --phase 1 --world-size 2 --epochs 5 --checkpoint-dir checkpoints

# Compute a Fisher payload from the phase-1 checkpoint
python compute_fisher.py --checkpoint checkpoints/latest.pt --output checkpoints/fisher.pt

# Phase 2: EWC + SWA refinement
python train.py --phase 2 --world-size 2 --swa-epochs 5 \
    --resume-from checkpoints/latest.pt --fisher-payload checkpoints/fisher.pt \
    --checkpoint-dir checkpoints

# Any of the above also runs on a GPU box by setting --world-size to the
# number of GPUs available
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
