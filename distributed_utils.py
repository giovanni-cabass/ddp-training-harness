"""Distributed-training robustness utilities.

This module holds the patterns that make long, unattended multi-GPU (or
CPU/gloo) training jobs survive the failure modes that matter in practice:
a rank that goes silent mid-collective (e.g. because a dtype/shape/NaN bug
that only shows up on that rank's data crashed it), a SIGTERM from a job
scheduler enforcing a walltime limit, and a crash mid-checkpoint. Nothing
here validates for a dtype/shape/NaN bug directly -- it makes the stall
that bug causes diagnosable (which rank, which collective) instead of an
indefinite hang. None of the functions below are specific to any
particular model or dataset.
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
    """Probe an ephemeral port and release it for the workers to bind to.

    A demo-only addition, not present in the real pipeline this
    generalizes: that system hardcodes nccl with a fixed, externally
    configured port, no runtime discovery at all. This exists purely so
    the demo can run standalone, repeatedly, without a scheduler assigning
    a port -- and it has an accepted, unmitigated TOCTOU race (another
    process could grab this exact port between the probe and the workers'
    actual bind). Deliberately left unmitigated rather than patched further:
    fixing it properly would mean retry/consensus machinery on top of an
    addition that already doesn't exist in the original.
    """
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
    # Bind the device before rendezvous, not after: NCCL communicator setup
    # can use the current device as part of init, so setting it only after
    # init_process_group (the previous ordering here) is less defensive than
    # necessary -- untested either way, since CI only ever exercises gloo.
    device_id = None
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device_id = torch.device("cuda", rank)
    hang_debug_log(rank, "pre init_process_group")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, device_id=device_id)
    hang_debug_log(rank, "post init_process_group")
    if backend == "nccl":
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
    global _MONITOR_GROUP
    if not dist.is_initialized():
        return
    rank = dist.get_rank()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as cuda_exc:
            print(f"Rank {rank}: CUDA sync/cache clear during teardown failed: {cuda_exc}", flush=True)
    monitored_barrier("teardown", rank, timeout_seconds=timeout_seconds)
    if _MONITOR_GROUP is not None:
        dist.destroy_process_group(_MONITOR_GROUP)
        _MONITOR_GROUP = None
    dist.destroy_process_group()


def install_graceful_shutdown_handler(on_signal: Callable[[], None]) -> None:
    """Register SIGTERM/SIGINT handlers that run `on_signal` once.

    A SLURM walltime kill sends SIGTERM before the hard SIGKILL. Resume
    always happens from whatever checkpoint was last saved when the
    selection metric improved -- this handler doesn't save an extra
    checkpoint on its own behalf, it just gives the training loop a chance
    to stop cleanly (finish the current step, tear down the process group)
    before the hard kill lands, instead of being killed mid-collective. A
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


@dataclass
class CheckpointHandoff:
    """Rank-0 crash-safe checkpoint writer.

    Writes to a temp path and renames into place with os.replace (atomic on
    POSIX), so a crash mid-write never leaves a corrupt checkpoint that a
    resumed job would load. Applied consistently to every checkpoint write
    here, broader than strictly necessary -- a deliberate strengthening of
    the pattern, not a claim that it matches any one real system verbatim.

    Two more differences from the real pipeline this generalizes: its own
    checkpoint writes save a bare state_dict with no wrapper, with step and
    metric bookkeeping kept in a separate external run log instead of
    inside the checkpoint file -- this wraps every save in a dict that
    embeds that bookkeeping directly. And it keeps only one "best" file
    (overwritten in place); this additionally keeps a uniquely numbered
    archival file on every improvement, a full history the real pipeline
    doesn't.
    """

    checkpoint_dir: Path

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict, step: int) -> None:
        # Each file's own write is atomic (temp + os.replace), but the pair
        # isn't jointly atomic -- a crash between the two leaves one of them
        # stale. Write latest.pt first: everything downstream (compute_fisher.py,
        # --resume-from, CI) reads latest.pt, so a crash in that gap should cost
        # the archival per-step file, never leave the actual resume point stale.
        for target_name in ("latest.pt", f"checkpoint_step{step:08d}.pt"):
            final_path = self.checkpoint_dir / target_name
            tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            try:
                torch.save(state, tmp_path)
                os.replace(tmp_path, final_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
