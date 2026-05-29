#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks-per-node=34
#SBATCH --mem=127000M
#SBATCH --time=40:00:00
#SBATCH --job-name=rgct_meta_omniglot

# W&B hyperparameter sweep agent for RGCT on Meta-Omniglot.
# Uses the already-created sweep:
#   https://wandb.ai/leathead_AQ_AM_IO/RGCT/sweeps/tfow3o6u
#
# Usage:
#   DATADIR=/path/to/meta-dataset/processed_data sbatch meta_omniglot_no_ctb_sweep.sh
#   DATA_DIR=/path/to/meta-dataset/processed_data bash meta_omniglot_no_ctb_sweep.sh

set -eo pipefail

# ============================ EDIT THESE ============================
DATASET="Meta-Omniglot"
SWEEP_ID="leathead_AQ_AM_IO/RGCT/tfow3o6u"

# Meta-Dataset records root. Keep this outside the sweep so Slurm jobs can
# change it per machine or allocation.
DATADIR="${RGCT_META_RECORDS_ROOT:-${DATADIR:-${DATA_DIR:-${RECORDS:-}}}}"

# Repo root. Submit from the repo root, or set REPO_DIR explicitly.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Optional virtualenv and Python override.
RGCT_ENV="${RGCT_ENV:-/home/ahmedm04/projects/DINOSEG/.venv}"
RGCT_PYTHON="${RGCT_PYTHON:-python}"

# Parallel W&B agents to launch (0 => one per visible GPU).
NUM_AGENTS="${NUM_AGENTS:-0}"
# ====================================================================

if [ -z "$DATADIR" ]; then
    echo "ERROR: set DATADIR, DATA_DIR, RECORDS, or RGCT_META_RECORDS_ROOT." >&2
    exit 1
fi

if [ -n "${SLURM_TMPDIR:-}" ]; then
    if command -v module >/dev/null 2>&1; then
        module load python/3.11 gcc cuda cudnn || true
    fi
fi

if [ -d "$RGCT_ENV" ]; then
    source "$RGCT_ENV/bin/activate"
fi

cd "$REPO_DIR"

export DATADIR
export DATA_DIR="$DATADIR"
export RECORDS="$DATADIR"
export RGCT_META_RECORDS_ROOT="$DATADIR"
export META_DATASET_ROOT="${META_DATASET_ROOT:-$REPO_DIR/meta-dataset}"
export RGCT_PYTHON

echo "Dataset  : $DATASET"
echo "Repo dir : $REPO_DIR"
echo "Data dir : $DATADIR"
echo "Sweep ID : $SWEEP_ID"
echo "Python   : $RGCT_PYTHON"

NUM_GPUS="$("$RGCT_PYTHON" -c 'import torch; print(max(1, torch.cuda.device_count()))' 2>/dev/null || echo 1)"
if [ "$NUM_AGENTS" -le 0 ]; then NUM_AGENTS="$NUM_GPUS"; fi
echo "Launching $NUM_AGENTS agent(s) across $NUM_GPUS GPU(s)."

pids=()
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    GPU=$((i % NUM_GPUS))
    echo "  agent $i -> GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" wandb agent "$SWEEP_ID" &
    pids+=($!)
done

wait "${pids[@]}"
echo "All agents for $DATASET finished."
