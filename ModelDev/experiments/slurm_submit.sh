#!/usr/bin/env bash
# Submit an immutable experiment manifest as a Slurm array. Logs go in ModelDev/logs.
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "Usage: slurm_submit.sh MANIFEST [MODELDEV_DIR] [MAX_CONCURRENT]" >&2; exit 2; fi
MANIFEST=$(realpath "$1")
MODELDEV_DIR=${2:-$(cd "$(dirname "$0")/.." && pwd)}
MAX_CONCURRENT=${3:-18}
LOG_DIR="$MODELDEV_DIR/logs"; mkdir -p "$LOG_DIR"
N=$(python - "$MANIFEST" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))["tasks"]))
PY
)
if [[ "$N" -eq 0 ]]; then echo "Manifest has no tasks: $MANIFEST" >&2; exit 2; fi
sbatch --array="0-$((N-1))%${MAX_CONCURRENT}" --output="$LOG_DIR/%x-%A_%a.out" --error="$LOG_DIR/%x-%A_%a.err" \
  "$MODELDEV_DIR/experiments/slurm_array_worker.sh" "$MANIFEST" "$MODELDEV_DIR"
