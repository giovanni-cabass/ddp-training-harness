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

# The space check below uses GNU-only du/df flags. On Linux (the real
# target: a SLURM cluster, or CI's ubuntu-latest) plain du/df already are
# the GNU versions. On macOS they're the BSD versions and don't support
# these flags -- but macOS's GNU coreutils (from Homebrew or MacPorts) are
# commonly installed as gdu/gdf precisely so they don't clobber the
# system's own du/df, so prefer those if present rather than requiring the
# caller to rearrange PATH. Verify it's actually GNU coreutils first: "gdu"
# also names an unrelated, independently popular disk-usage browser, and a
# false match there would fail confusingly rather than falling through.
pick_gnu_tool() {
  local candidate
  candidate="$(command -v "$1" 2>/dev/null || true)"
  if [[ -n "$candidate" ]] && "$candidate" --version 2>/dev/null | grep -qi "GNU coreutils"; then
    echo "$candidate"
  else
    command -v "$2"
  fi
}
DU_CMD="$(pick_gnu_tool gdu du)"
DF_CMD="$(pick_gnu_tool gdf df)"

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
    # Guarded with `|| true`: this runs from an EXIT trap under `set -e`, and a
    # trap's own failing last command overrides the exit status that triggered
    # it -- a transient rm failure (busy mount, stale handle) must never mask
    # a real training success or a more informative earlier failure.
    rm -rf "$SCRATCH_BASE" || true
  fi
}
trap cleanup_on_exit EXIT

mkdir -p "$SCRATCH_BASE"
# Mark staged as of here, not after rsync succeeds below: anything that fails
# from this point on (synthetic-shard materialization, the space check,
# rsync itself) has still created real content under $SCRATCH_BASE that must
# be cleaned up on exit, not left behind because STAGE_DONE never flipped.
STAGE_DONE=1

if [[ -z "$SRC_DATA_DIR" ]]; then
  echo "SRC_DATA_DIR not set -- materializing a synthetic shard to stand in for a real export."
  SRC_DATA_DIR="$SCRATCH_BASE/source"
  python data.py --output-dir "$SRC_DATA_DIR"
fi

echo "Checking available scratch space..."
src_bytes="$("$DU_CMD" -sb "$SRC_DATA_DIR" | awk '{print $1}')"
avail_bytes="$("$DF_CMD" -B1 --output=avail "$SCRATCH_BASE" | awk 'NR==2 {print $1}')"
if (( src_bytes > avail_bytes )); then
  echo "ERROR: not enough scratch space (need $src_bytes bytes, have $avail_bytes)." >&2
  exit 2
fi
echo "Space check OK."

echo "Staging $SRC_DATA_DIR -> $STAGE_DIR"
rsync -a "$SRC_DATA_DIR/" "$STAGE_DIR/"

echo "Running training against staged data..."
python train.py --data-dir "$STAGE_DIR" "$@"

echo "Training finished; cleanup will run via the EXIT trap."
