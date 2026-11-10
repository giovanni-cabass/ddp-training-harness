# DDP Training Harness

[![CI](https://github.com/giovanni-cabass/ddp-training-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/giovanni-cabass/ddp-training-harness/actions/workflows/ci.yml)

*A from-scratch demo of DDP training-infra patterns — toy model, synthetic
data, no production code.*

Small, dependency-light patterns for running long, unattended multi-GPU
training jobs on a shared cluster with PyTorch's DDP (DistributedDataParallel)
— built to survive the failure modes that actually show up in practice, not
just the happy path.

## Why this exists

On a shared SLURM-style cluster, a training job usually doesn't end because
training finished — it ends because the walltime budget ran out. I lean into
that constraint rather than fighting it: instead of early-stopping on a
patience window, I estimate epochs-per-hour for a given architecture/dataset
by hand, offline, before submitting the job, and budget walltime accordingly
— no script computes this for me. The job runs until the scheduler kills
it, and the next one resumes from a checkpoint. Early stopping has a
sharper failure mode too, in my experience: a loss curve can plateau, look
converged, and then undergo a genuine transition to a lower plateau later on
— a patience window tuned for the first plateau stops before ever seeing
the second.

This resume-after-walltime-kill approach only works if the harness can
(a) tell when something has gone quietly wrong on one rank instead of
hanging forever, and (b) not lose the run to the walltime kill. Checkpointing
every time the selection metric improves is what actually makes the run
recoverable; the `SIGTERM` signal SLURM sends before the hard kill triggers
a handler that doesn't save anything on its own — it just lets the current
step finish and the process group tear down cleanly instead of being killed
mid-collective. (That handler relies on the OS delivering the signal
directly to each worker process, true under a SLURM job kill; a `SIGTERM`
sent only to the top-level `python train.py` process wouldn't reach the
workers it spawns.)

The same shared-cluster reality shapes how data gets to the job, too:
`stage_and_run.sh` copies input data to node-local scratch before training
starts and cleans it up on any exit, so a multi-epoch job doesn't hammer a
shared filesystem with repeated random-access reads, and nothing lingers on
a compute node other people also use.

## What's inside

| File | Pattern |
|---|---|
| `distributed_utils.py` | process-group robustness, in one module:<br>- **monitored barrier**, backed by a side-channel `gloo` group even when the main backend is `nccl` (`nccl`'s own collectives can hang indefinitely on a stuck rank; `gloo`'s `monitored_barrier` supports a timeout and names the unresponsive rank)<br>- per-rank timestamped hang-debug logging around process-group setup and the monitored barrier<br>- `SIGTERM`/`SIGINT` handlers that run a shutdown hook exactly once<br>- a rank-0 checkpoint writer that's crash-safe (temp file + atomic rename) |
| `continual.py` | Fisher/EWC/SWA continual-learning refinements, in one module:<br>- a diagonal Fisher-information estimate<br>- a provenance-tagged Fisher payload (a later run refuses to load one computed from the wrong checkpoint)<br>- an EWC (Elastic Weight Consolidation) penalty anchored to a reference checkpoint, with the Fisher diagonal clipped at a percentile and mean-normalized first (so a few outlier parameters can't dominate the penalty, and `--ewc-importance` means roughly the same magnitude of penalty run to run)<br>- an SWA (Stochastic Weight Averaging) tracker that keeps both the running weight-average *and* the best single snapshot by selection metric, and ships whichever one actually wins rather than assuming the average is better by construction |
| `compute_fisher.py` | standalone, single-process script: load a checkpoint, verify it's fp32, compute the Fisher diagonal on the train-only split, save a provenance-tagged payload — not distributed, since it's a one-shot post-training pass, not a training job. Skips recomputing if the output already matches the checkpoint's own fingerprint, unless `--overwrite` forces it |
| `model.py` / `data.py` | a toy MLP regressor and a synthetic dataset — stand-ins for the real job this pattern set was built for |
| `stage_and_run.sh` | node-local scratch staging: pre-flight disk-space check (with a GNU-vs-BSD coreutils fallback so the same script runs on both a Linux cluster and macOS dev), `rsync` from shared storage to scratch, `trap ... EXIT`-based cleanup that runs on success/failure/`SIGTERM` alike, and a safety guard refusing to operate on anything that doesn't look like disposable scratch |
| `train.py` | two phases, spawned via `torch.multiprocessing.spawn` (`nccl` if GPUs are visible, `gloo` fallback on CPU) rather than an external launcher like `torchrun`:<br>- **Phase 1**: DDP training with a genuine three-way split — a *selection* set used only to pick the best checkpoint, and a *held-out* exam set that never influences any model-selection decision<br>- **Phase 2**: resumes from a phase-1 checkpoint and runs a long constant-LR SWA window; optionally adds an EWC penalty too, if a Fisher payload (from `compute_fisher.py`) is given — a plain SWA-only continuation is also a valid run<br>- gradient microbatching via `no_sync()` accumulation, to simulate a larger effective batch size under a memory ceiling — DDP gradients sync only on the final microbatch of each step<br>- gradient-norm clipping (`--grad-clip-norm`, default 1.0), applied after microbatch accumulation and before the optimizer step, in both phases |

## Running it

```bash
pip install -r requirements.txt

# Phase 1: train from scratch (CPU, gloo, 2 simulated ranks)
python train.py --phase 1 --world-size 2 --epochs 5 --checkpoint-dir checkpoints

# Same thing, via the node-local staging wrapper (what actually runs on a
# cluster -- stages data to scratch first, cleans up on any exit)
./stage_and_run.sh --phase 1 --world-size 2 --epochs 5 --checkpoint-dir checkpoints

# If phase 1 got walltime-killed, restart it from the last improved
# checkpoint explicitly -- there's no auto-resume: as in the real workflow
# this generalizes, you point the next job at a specific checkpoint path.
python train.py --phase 1 --world-size 2 --epochs 5 --checkpoint-dir checkpoints \
    --resume-from checkpoints/latest.pt

# Compute a Fisher payload from the phase-1 checkpoint
python compute_fisher.py --checkpoint checkpoints/latest.pt --output checkpoints/fisher.pt

# Phase 2: EWC + SWA refinement
python train.py --phase 2 --world-size 2 --swa-epochs 5 \
    --resume-from checkpoints/latest.pt --fisher-payload checkpoints/fisher.pt \
    --checkpoint-dir checkpoints

# On a GPU box this switches to the nccl backend automatically (based on
# CUDA availability); set --world-size to match the number of GPUs you want to use
```

## Tests / CI

```bash
python -m pytest -v
```

Tests run CPU-only (the DDP ones over `gloo`) so they don't need GPU
hardware — including real phase-1-resume and phase-2 runs through `train.py`
itself, not mocked calls. CI runs the suite on every push, then goes further
still, as a separate smoke-test layer: `stage_and_run.sh`, `compute_fisher.py`,
and phase 2 all run for real, end-to-end, against a freshly materialized
synthetic shard — so the whole pipeline is verified green on every push, not
just isolated functions.

## Status

Core patterns are implemented: process-group setup/teardown with a monitored
barrier, graceful `SIGTERM`/`SIGINT` shutdown, crash-safe checkpointing,
gradient microbatching, and the Fisher/EWC/SWA pipeline across `train.py` +
`compute_fisher.py`. CI and the test suite only ever exercise the CPU/`gloo`
path; the `nccl`/GPU path is implemented but not CI-verified.

## License

MIT, see `LICENSE`.
