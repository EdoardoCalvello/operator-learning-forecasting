#!/bin/bash
# KSE smoothing training. Run directly (bash) or submit with sbatch from the repository root.
# Usage: run.sh [config, default configs/kse_smoothing.yaml] [extra arguments for kse_smoothing.py]
# Add your cluster's account/partition/constraint options to the #SBATCH lines or the sbatch command.
#SBATCH --job-name=kse_smoothing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=09:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
# sbatch runs a copy of this file; locate the checkout from the submission directory.
REPO_ROOT="${TNO_REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" rev-parse --show-toplevel)}"
config=${1:-configs/kse_smoothing.yaml}
shift $(( $# > 0 ? 1 : 0 ))
exec bash "$REPO_ROOT/scripts/launch.sh" "scripts/KSE/smoothing/kse_smoothing.py" "$config" "$@"
