"""Distributed-training robustness utilities.

This module holds the patterns that make long, unattended multi-GPU (or
CPU/gloo) training jobs survive the failure modes that matter in practice:
a stalled rank, a dtype/shape/NaN problem visible on only one rank, a
SIGTERM from a job scheduler enforcing a walltime limit, and a crash
mid-checkpoint. None of the functions below are specific to any particular
model or dataset.
"""

from __future__ import annotations

import os
import signal
import socket
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist

_MONITOR_GROUP: Optional[dist.ProcessGroup] = None


def resolve_backend() -> str:
    """Return "nccl" if CUDA is visible, else "gloo"."""
    return "nccl" if torch.cuda.is_available() else "gloo"


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def hang_debug_log(rank: int, stage: str) -> None:
    """Per-rank timestamped log line around collective calls.

    Cheap but high-value: when a job hangs, these lines are what tell you
    *which* rank, at *which* collective, stopped responding.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[rank {rank}] {timestamp} {stage}", flush=True)


def setup_process_group(rank: int, world_size: int, backend: str, master_port: int) -> None:
    """Initialize this rank's process group.

    If the main backend is nccl, also stands up a side-channel gloo group
    purely for monitored_barrier() -- nccl's own collectives can hang
    indefinitely on a stuck rank, while gloo's monitored_barrier supports a
    timeout and names the unresponsive rank(s).
    """
    global _MONITOR_GROUP
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    hang_debug_log(rank, "pre init_process_group")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    hang_debug_log(rank, "post init_process_group")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        _MONITOR_GROUP = dist.new_group(backend="gloo")
    else:
        _MONITOR_GROUP = None  # default group is already gloo; it supports monitored_barrier directly


def monitored_barrier(stage: str, rank: int, timeout_seconds: float = 30.0) -> bool:
    """Barrier with a timeout. Never raises -- returns False if a rank is stuck."""
    hang_debug_log(rank, f"pre monitored_barrier:{stage}")
    try:
        dist.monitored_barrier(group=_MONITOR_GROUP, timeout=timedelta(seconds=timeout_seconds))
    except Exception as exc:  # deliberately broad: any failure here means "don't hang the job"
        print(f"[rank {rank}] monitored_barrier at '{stage}' failed/timed out: {exc}", flush=True)
        return False
    hang_debug_log(rank, f"post monitored_barrier:{stage}")
    return True


def teardown_process_group(timeout_seconds: float = 30.0) -> None:
    """Tear down this rank's process group.

    Uses monitored_barrier (not a plain barrier) so a stuck peer doesn't hang
    teardown forever -- after the timeout we destroy the process group
    regardless of whether every rank actually reached it.
    """
    if not dist.is_initialized():
        return
    rank = dist.get_rank()
    monitored_barrier("teardown", rank, timeout_seconds=timeout_seconds)
    dist.destroy_process_group()


def install_graceful_shutdown_handler(on_signal: Callable[[], None]) -> None:
    """Register SIGTERM/SIGINT handlers that run `on_signal` once.

    A SLURM walltime kill sends SIGTERM before the hard SIGKILL. The primary
    resume point is whatever checkpoint was last saved when the selection
    metric improved; this handler adds one extra checkpoint-and-clean-teardown
    attempt as a safety net for the rarer case where the kill lands between
    improvement checks, rather than losing that partial epoch outright. A
    second signal is not re-handled -- by then we're already on our way out.
    """
    handled = {"done": False}

    def _handler(signum: int, _frame: object) -> None:
        if handled["done"]:
            return
        handled["done"] = True
        print(f"[signal] received {signal.Signals(signum).name}, running shutdown hook", flush=True)
        on_signal()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def broadcast_exit_on_mismatch(local_ok: bool, rank: int, device: torch.device) -> bool:
    """All-reduce a per-rank health flag; return True iff every rank is healthy.

    Used to catch a dtype/shape/NaN problem that only one rank can see (e.g.
    a corrupt shard, a bad batch) and have every rank exit together, instead
    of the healthy ranks deadlocking on the next collective waiting for a
    peer that already bailed out.
    """
    flag = torch.tensor([0 if local_ok else 1], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    any_unhealthy = bool(flag.item() > 0)
    if any_unhealthy and local_ok:
        print(f"[rank {rank}] another rank reported unhealthy; exiting together", flush=True)
    return not any_unhealthy


@dataclass
class CheckpointHandoff:
    """Rank-0 crash-safe checkpoint writer.

    Writes to a temp path and renames into place with os.replace (atomic on
    POSIX), so a crash mid-write never leaves a corrupt checkpoint that a
    resumed job would load.
    """

    checkpoint_dir: Path

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict, step: int) -> None:
        for target_name in (f"checkpoint_step{step:08d}.pt", "latest.pt"):
            final_path = self.checkpoint_dir / target_name
            tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            try:
                torch.save(state, tmp_path)
                os.replace(tmp_path, final_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

    def load_latest(self) -> Optional[dict]:
        latest_path = self.checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            return None
        return torch.load(latest_path, map_location="cpu", weights_only=True)
