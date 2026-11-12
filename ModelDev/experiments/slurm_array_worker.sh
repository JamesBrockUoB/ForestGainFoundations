#!/usr/bin/env bash
# Generic Slurm array worker: one immutable manifest task per array index.
#SBATCH --job-name=fg-experiment
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/experiment-%A_%a.out
#SBATCH --error=logs/experiment-%A_%a.err
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: slurm_array_worker.sh MANIFEST [MODELDEV_DIR]" >&2
  exit 2
fi
MANIFEST=$(realpath "$1")
MODELDEV_DIR=${2:-$(cd "$(dirname "$0")/.." && pwd)}
TASK_ID=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}
mkdir -p "$MODELDEV_DIR/logs"
cd "$MODELDEV_DIR"
python -m experiments run "$MANIFEST" --task "$TASK_ID" --cwd "$MODELDEV_DIR" --log-dir "$MODELDEV_DIR/logs"
