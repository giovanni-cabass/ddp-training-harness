"""Entry point for the DDP training harness demo.

Two phases, matching the real workflow this harness is modeled on:
  phase 1 -- DDP training from scratch, selecting the best checkpoint on a
             held-out *selection* set, while a second *exam* set is never
             touched by any model-selection decision;
  phase 2 -- resumes from a phase-1 checkpoint, adds an EWC penalty anchored
             to that checkpoint (using a Fisher payload from
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
    broadcast_exit_on_mismatch,
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
    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    checkpoints = CheckpointHandoff(args.checkpoint_dir)
    best_selection_loss = float("inf")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        local_ok = True
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
                local_ok = False
                break
            microbatched_step(ddp_model, optimizer, loss_fn, inputs, targets, args.microbatch_size)

        healthy = broadcast_exit_on_mismatch(local_ok, rank, device)
        if not healthy:
            if rank == 0:
                print(f"[epoch {epoch}] a rank reported bad data; aborting all ranks together", flush=True)
            break

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

        if shutdown_requested["flag"]:
            if rank == 0:
                print("[shutdown] saving checkpoint and exiting before the hard kill", flush=True)
                checkpoints.save(
                    {
                        "model_state": ddp_model.module.state_dict(),
                        "step": epoch,
                        "selection_loss": best_selection_loss,
                    },
                    step=epoch,
                )
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

    ewc = None
    if args.fisher_payload is not None:
        fingerprint = checkpoint_fingerprint(checkpoint["model_state"])
        payload = FisherPayload.load_matching(args.fisher_payload, expected_fingerprint=fingerprint)
        fisher_diag = {k: v.to(device) for k, v in payload.fisher_diag.items()}
        ewc = ElasticWeightConsolidation(model, fisher_diag, importance=args.ewc_importance)

    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.swa_lr)
    loss_fn = nn.MSELoss()

    swa_state = SwaState(model=ddp_model.module) if rank == 0 else None
    checkpoints = CheckpointHandoff(Path(args.checkpoint_dir) / "swa")

    for epoch in range(args.swa_epochs):
        sampler.set_epoch(epoch)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            microbatched_step(ddp_model, optimizer, loss_fn, inputs, targets, args.microbatch_size, ewc=ewc)

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
    args = parser.parse_args()

    if args.phase == 2 and args.resume_from is None:
        parser.error("--resume-from is required for --phase 2")

    backend = resolve_backend()
    master_port = find_free_port()
    mp.spawn(worker, args=(args.world_size, backend, master_port, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
