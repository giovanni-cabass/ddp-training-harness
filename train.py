"""Entry point for the DDP training harness demo.

Two phases:
  phase 1 -- the core DDP harness this repo demonstrates: training from
             scratch, selecting the best checkpoint on a held-out
             *selection* set, while a second *exam* set is never touched by
             any model-selection decision;
  phase 2 -- bonus content layered on top (see continual.py), showing a
             common follow-up to a converged checkpoint rather than
             retraining from scratch: resumes from a phase-1 checkpoint,
             adds an EWC penalty anchored to it (using a Fisher payload from
             compute_fisher.py) and a long constant-LR "SWA" window,
             tracking both the running weight-average and the best single
             snapshot by selection metric, shipping whichever one actually
             wins.

Spawns workers itself via torch.multiprocessing.spawn -- nccl if GPUs are
visible, gloo fallback on CPU -- rather than relying on a torchrun-style
external launcher.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from continual import ElasticWeightConsolidation, FisherPayload, SwaState, checkpoint_fingerprint
from data import load_dataset
from distributed_utils import (
    CheckpointHandoff,
    find_free_port,
    install_graceful_shutdown_handler,
    monitored_barrier,
    resolve_backend,
    setup_process_group,
    teardown_process_group,
)
from model import ToyRegressor


def split_dataset(dataset, train_frac: float = 0.8, selection_frac: float = 0.1):
    """Three-way split: train / selection (picks the best checkpoint) / held-out
    exam set that never influences any model-selection decision.
    """
    n = len(dataset)
    n_train = int(n * train_frac)
    n_selection = int(n * selection_frac)
    indices = list(range(n))
    train_idx = indices[:n_train]
    selection_idx = indices[n_train : n_train + n_selection]
    heldout_idx = indices[n_train + n_selection :]
    return Subset(dataset, train_idx), Subset(dataset, selection_idx), Subset(dataset, heldout_idx)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> float:
    """Run rank 0 alone, over the full (unsharded) selection set.

    A simplification: the real pipeline this generalizes shards the
    selection data across ranks and all-reduces the metric, so every rank
    ends up with the same already-synchronized value. At this toy scale,
    evaluating only on rank 0 is equivalent and simpler, not a correctness
    shortcut.
    """
    model.eval()
    total_loss, total_count = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        total_loss += loss.item() * inputs.size(0)
        total_count += inputs.size(0)
    model.train()
    return total_loss / max(total_count, 1)


def microbatched_step(
    ddp_model: DDP,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    microbatch_size: int,
    ewc: Optional[ElasticWeightConsolidation] = None,
    grad_clip_norm: Optional[float] = None,
) -> float:
    """Accumulate gradients over microbatches, only syncing DDP gradients on the last one."""
    optimizer.zero_grad(set_to_none=True)
    batch_size = inputs.size(0)
    num_microbatches = max(1, (batch_size + microbatch_size - 1) // microbatch_size)
    total_loss = 0.0

    for i in range(num_microbatches):
        start, end = i * microbatch_size, min((i + 1) * microbatch_size, batch_size)
        micro_inputs, micro_targets = inputs[start:end], targets[start:end]
        is_last = i == num_microbatches - 1

        sync_context = nullcontext() if is_last else ddp_model.no_sync()
        with sync_context:
            predictions = ddp_model(micro_inputs)
            loss = loss_fn(predictions, micro_targets) * (micro_inputs.size(0) / batch_size)
            if ewc is not None and is_last:
                loss = loss + ewc.penalty()
            loss.backward()
        total_loss += loss.item()

    # The real pipeline this generalizes clips gradients after accumulation,
    # before stepping; this had been omitted here until flagged -- the
    # clip value itself is a generic default, not the real threshold.
    if grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip_norm)
    optimizer.step()
    return total_loss


def run_phase1(rank: int, world_size: int, device: torch.device, args: argparse.Namespace, shutdown_requested: dict) -> None:
    dataset = load_dataset(args.data_dir, seed=0)
    train_set, selection_set, heldout_set = split_dataset(dataset)

    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    selection_loader = DataLoader(selection_set, batch_size=args.batch_size)
    heldout_loader = DataLoader(heldout_set, batch_size=args.batch_size)

    model = ToyRegressor().to(device)
    best_selection_loss = float("inf")
    start_epoch = 0
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        # Carry the prior best forward too -- otherwise the first evaluated
        # epoch after resuming would always look like "an improvement" and
        # overwrite a genuinely better checkpoint with a worse one.
        best_selection_loss = checkpoint.get("selection_loss", float("inf"))
        # Continue epoch numbering from the checkpoint's own step, not from 0
        # -- step is what CheckpointHandoff.save() uses to name the archival
        # checkpoint_step*.pt file, so restarting at 0 would silently
        # overwrite the previous run's archival files with unrelated
        # content instead of giving each epoch, across the whole resume
        # history, its own distinct file.
        start_epoch = checkpoint.get("step", -1) + 1
        if rank == 0:
            print(
                f"Resumed phase 1 from {args.resume_from} "
                f"(selection loss so far: {best_selection_loss:.4f}, continuing from epoch {start_epoch})",
                flush=True,
            )

    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    # Optimizer state (Adam's momentum/variance buffers) is never saved or
    # restored on resume, here or in the real pipeline this generalizes --
    # a resumed run always starts from a freshly constructed optimizer,
    # cold.
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    checkpoints = CheckpointHandoff(args.checkpoint_dir)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        sampler.set_epoch(epoch)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            microbatched_step(
                ddp_model, optimizer, loss_fn, inputs, targets, args.microbatch_size,
                grad_clip_norm=args.grad_clip_norm,
            )

        # Called every epoch here; the real workflow this generalizes only
        # uses its gloo side-channel barrier once, at final teardown -- an
        # addition, though a cheap and safe one (a timeout-bounded check,
        # not a decision point that can itself diverge ranks).
        monitored_barrier(f"epoch-{epoch}-end", rank)

        if rank == 0:
            selection_loss = evaluate(ddp_model.module, selection_loader, loss_fn, device)
            print(f"[epoch {epoch}] selection loss = {selection_loss:.4f}", flush=True)
            if selection_loss < best_selection_loss:
                best_selection_loss = selection_loss
                checkpoints.save(
                    {"model_state": ddp_model.module.state_dict(), "step": epoch, "selection_loss": selection_loss},
                    step=epoch,
                )

        # shutdown_requested is set per-rank, from that rank's own OS signal
        # delivery, with no cross-rank broadcast of the decision -- if one
        # rank's SIGTERM is delivered or handled late relative to another's,
        # that rank could proceed into the next epoch's collective while its
        # peer tears down. Not a demo-introduced gap: the real pipeline this
        # generalizes has the same limitation (each rank's signal handling
        # is independent there too, with no cross-rank consensus on when to
        # stop). Accepted as a known, extremely low-probability race in
        # both: under a real SLURM walltime kill, SIGTERM goes to the whole
        # process group at effectively the same instant, and fixing it here
        # would mean adding a per-epoch cross-rank consensus check of
        # exactly the kind this repo deliberately removed elsewhere for
        # being more complexity than it was worth.
        if shutdown_requested["flag"]:
            if rank == 0:
                print("[shutdown] exiting cleanly before the hard kill -- last improvement was already checkpointed", flush=True)
            break

    if rank == 0:
        held_out_loss = evaluate(ddp_model.module, heldout_loader, loss_fn, device)
        print(f"Final held-out exam loss (never used for selection): {held_out_loss:.4f}", flush=True)


def run_phase2(rank: int, world_size: int, device: torch.device, args: argparse.Namespace, shutdown_requested: dict) -> None:
    dataset = load_dataset(args.data_dir, seed=0)
    train_set, selection_set, _heldout_set = split_dataset(dataset)

    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    selection_loader = DataLoader(selection_set, batch_size=args.batch_size)

    model = ToyRegressor().to(device)
    checkpoint = torch.load(args.resume_from, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    # Optional here (a plain SWA-only run is allowed with no Fisher
    # payload); in the real dedicated EWC+SWA script this generalizes, EWC
    # is mandatory, not an optional extra.
    ewc = None
    if args.fisher_payload is not None:
        fingerprint = checkpoint_fingerprint(checkpoint["model_state"])
        payload = FisherPayload.load_matching(args.fisher_payload, expected_fingerprint=fingerprint)
        fisher_diag = {k: v.to(device) for k, v in payload.fisher_diag.items()}
        ewc = ElasticWeightConsolidation(
            model, fisher_diag, importance=args.ewc_importance, clip_percentile=args.ewc_fisher_clip_percentile
        )

    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    # Constant LR and averaging from the very first epoch here. The real
    # pipeline this generalizes runs one longer continuation window that
    # starts with a non-constant warmup schedule before switching to the
    # constant SWA LR and only then beginning to average -- compressed away
    # here for a toy demo rather than misrepresented as identical.
    # Optimizer choice is also a deviation, not just the schedule: the real
    # pipeline uses AdamW throughout, in phase 1 and this continuation
    # alike, and never switches optimizer family. This demo uses Adam in
    # phase 1 (see run_phase1) and switches to plain SGD here.
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.swa_lr)
    loss_fn = nn.MSELoss()

    swa_state = SwaState(model=ddp_model.module) if rank == 0 else None
    if swa_state is not None:
        # Seed "best" with the phase-1 checkpoint's own recorded loss, not
        # just infinity: the real pipeline this generalizes tracks "best"
        # continuously across the whole run (pre-SWA and SWA epochs alike),
        # so it can never ship something worse than what it started from.
        # Phase 1 and phase 2 here are separate invocations with otherwise
        # no shared "best" state -- without this, pick_shipping_state()
        # could ship a phase-2 result that's actually a regression against
        # the phase-1 checkpoint it resumed from.
        swa_state.consider_best(checkpoint.get("selection_loss", float("inf")))
    checkpoints = CheckpointHandoff(Path(args.checkpoint_dir) / "swa")

    for epoch in range(args.swa_epochs):
        sampler.set_epoch(epoch)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            microbatched_step(
                ddp_model, optimizer, loss_fn, inputs, targets, args.microbatch_size,
                ewc=ewc, grad_clip_norm=args.grad_clip_norm,
            )

        monitored_barrier(f"swa-epoch-{epoch}-end", rank)

        if rank == 0:
            selection_loss = evaluate(ddp_model.module, selection_loader, loss_fn, device)
            swa_state.update_average()
            improved = swa_state.consider_best(selection_loss)
            suffix = " (new best)" if improved else ""
            print(f"[swa epoch {epoch}] selection loss = {selection_loss:.4f}{suffix}", flush=True)

        if shutdown_requested["flag"]:
            break

    if rank == 0:

        def _evaluate_state(state_dict: dict) -> float:
            model.load_state_dict(state_dict)
            return evaluate(model, selection_loader, loss_fn, device)

        shipping_state = swa_state.pick_shipping_state(_evaluate_state)
        checkpoints.save({"model_state": shipping_state}, step=args.swa_epochs)
        print(
            "Saved SWA-phase shipping checkpoint "
            "(best-by-selection-metric vs. running average, whichever won).",
            flush=True,
        )


def worker(rank: int, world_size: int, backend: str, master_port: int, args: argparse.Namespace) -> None:
    setup_process_group(rank, world_size, backend, master_port)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")

    # Relies on the OS delivering SIGTERM directly to this rank's own
    # process, not relayed by the parent -- true under a SLURM walltime
    # kill (the whole process group is signaled), but a SIGTERM sent only
    # to the top-level `python train.py` process (mp.spawn's parent), not
    # its rank children, would do nothing here.
    shutdown_requested = {"flag": False}
    install_graceful_shutdown_handler(lambda: shutdown_requested.update(flag=True))

    try:
        if args.phase == 1:
            run_phase1(rank, world_size, device, args, shutdown_requested)
        else:
            run_phase2(rank, world_size, device, args, shutdown_requested)
    finally:
        teardown_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP training harness demo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--swa-epochs", type=int, default=5)
    parser.add_argument("--swa-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--fisher-payload", type=Path, default=None)
    parser.add_argument("--ewc-importance", type=float, default=1.0)
    parser.add_argument("--ewc-fisher-clip-percentile", type=float, default=99.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    args = parser.parse_args()

    if args.phase == 2 and args.resume_from is None:
        parser.error("--resume-from is required for --phase 2")

    backend = resolve_backend()
    master_port = find_free_port()
    mp.spawn(worker, args=(args.world_size, backend, master_port, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
