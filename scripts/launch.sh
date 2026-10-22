#!/bin/bash
# Run an experiment entry point in its own output directory; invoked by the experiment run.sh files.
# Usage: bash scripts/launch.sh <entry.py> <config.yaml> [extra arguments for the entry point]
# Works locally and inside a Slurm job. Paths are relative to the repository root unless absolute.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
entry=$1
config=$2
shift 2
[[ $entry = /* ]] || entry="$REPO_ROOT/$entry"
[[ $config = /* ]] || config="$REPO_ROOT/$config"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MPLBACKEND=Agg
export TNO_DATA_DIR=${TNO_DATA_DIR:-$REPO_ROOT/data}
export TNO_OUTPUT_ROOT=${TNO_OUTPUT_ROOT:-$REPO_ROOT/outputs}
# One directory per run: checkpoints, W&B files and plots are written there.
# TNO_RUN_TAG separates concurrent runs that share one Slurm job.
run_id=${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}
export TNO_RUN_DIR="$TNO_OUTPUT_ROOT/$(basename "$entry" .py)/${run_id}${TNO_RUN_TAG:+_$TNO_RUN_TAG}"
mkdir -p "$TNO_RUN_DIR"
export WANDB_DIR="$TNO_RUN_DIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "Config: $config"
echo "Output: $TNO_RUN_DIR"
cd "$TNO_RUN_DIR"
exec python "$entry" --config "$config" "$@"
