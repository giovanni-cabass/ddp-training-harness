#!/bin/bash -l
# Stage input data to node-local scratch, run training against the staged
# copy, and clean up on any exit -- success, failure, or a scheduler's
# walltime SIGTERM.
#
# Mirrors a real pattern from training on a shared cluster: copying data to
# node-local fast storage before a multi-epoch job avoids hammering a slow
# shared filesystem with repeated random-access reads, and cleaning up
# afterward -- regardless of how the job ends -- keeps node-local scratch
# from leaking across jobs on a machine other people also use.
#
# Env vars (all optional):
#   SRC_DATA_DIR   where the prepared data already lives (shared storage).
#                  If unset, a synthetic shard is materialized to stand in
#                  for a real export.
#   SCRATCH_BASE   node-local scratch root. Defaults to $SLURM_TMPDIR if
#                  set (the scheduler's own node-local scratch), else a
#                  job-specific directory under /tmp.
#
# Any extra args are forwarded to train.py, e.g.:
#   ./stage_and_run.sh --phase 1 --world-size 2 --epochs 5

set -eo pipefail

JOB_ID="${SLURM_JOB_ID:-$$}"
SCRATCH_BASE="${SCRATCH_BASE:-${SLURM_TMPDIR:-/tmp/$USER/ddp_harness_$JOB_ID}}"
STAGE_DIR="$SCRATCH_BASE/data"
SRC_DATA_DIR="${SRC_DATA_DIR:-}"

# Refuse to stage into, or clean up, anything that doesn't look like
# disposable scratch space. A misconfigured or unset scratch variable
# should fail loudly, never silently operate on a persistent directory.
case "$SCRATCH_BASE" in
  "$HOME"|"$HOME"/*|/|"")
    echo "ERROR: refusing to use SCRATCH_BASE='$SCRATCH_BASE' -- doesn't look like disposable scratch." >&2
    exit 2
    ;;
esac

STAGE_DONE=0
cleanup_on_exit() {
  if [[ "$STAGE_DONE" -eq 1 ]]; then
    echo "Cleaning up scratch: $SCRATCH_BASE"
    rm -rf "$SCRATCH_BASE"
  fi
}
trap cleanup_on_exit EXIT

mkdir -p "$SCRATCH_BASE"

if [[ -z "$SRC_DATA_DIR" ]]; then
  echo "SRC_DATA_DIR not set -- materializing a synthetic shard to stand in for a real export."
  SRC_DATA_DIR="$SCRATCH_BASE/source"
  python data.py --output-dir "$SRC_DATA_DIR"
fi

echo "Checking available scratch space..."
src_bytes="$(du -sb "$SRC_DATA_DIR" | awk '{print $1}')"
avail_bytes="$(df -B1 --output=avail "$SCRATCH_BASE" | awk 'NR==2 {print $1}')"
if (( src_bytes > avail_bytes )); then
  echo "ERROR: not enough scratch space (need $src_bytes bytes, have $avail_bytes)." >&2
  exit 2
fi
echo "Space check OK."

echo "Staging $SRC_DATA_DIR -> $STAGE_DIR"
rsync -a --info=progress2 "$SRC_DATA_DIR/" "$STAGE_DIR/"
STAGE_DONE=1

echo "Running training against staged data..."
python train.py --data-dir "$STAGE_DIR" "$@"

echo "Training finished; cleanup will run via the EXIT trap."
