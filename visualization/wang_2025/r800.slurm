#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    : "${ACCOUNT:?set ACCOUNT to the GPU allocation account}"
    : "${VISUALIZATION_OUTPUT_DIR:?set VISUALIZATION_OUTPUT_DIR to a new directory in project storage}"

    PARTITION="${PARTITION:-gpu}"
    CPUS="${CPUS:-72}"
    WALLTIME="${WALLTIME:-01:00:00}"
    GPU_ENV="${TOPAX_GPU_ENV:-$HOME/envs/topax-gpu.sh}"
    [[ -f "$GPU_ENV" ]] || {
        echo "GPU environment script not found: $GPU_ENV" >&2
        exit 1
    }
    LOG_DIR="$TOPAX_ROOT/visualization/.slurm"
    mkdir -p "$LOG_DIR"

    job_id="$(
        sbatch --parsable \
            -A "$ACCOUNT" -p "$PARTITION" -N 1 --ntasks=1 \
            --cpus-per-task="$CPUS" --gpus=1 -t "$WALLTIME" \
            -J topax-wang-r800-visualization \
            -o "$LOG_DIR/wang-r800-%j.out" \
            --export=ALL,TOPAX_ROOT="$TOPAX_ROOT",VISUALIZATION_OUTPUT_DIR="$VISUALIZATION_OUTPUT_DIR",TOPAX_GPU_ENV="$GPU_ENV" \
            "$0"
    )"
    echo "Submitted Wang R800 visualization job: $job_id"
    exit 0
fi

# shellcheck disable=SC1090
source "${TOPAX_GPU_ENV:?}"
export LC_ALL=C
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    if [[ " ${XLA_FLAGS:-} " != *" --xla_gpu_cuda_data_dir="* ]]; then
        export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=$CUDA_HOME"
    fi
fi
unset XLA_PYTHON_CLIENT_ALLOCATOR XLA_CLIENT_MEM_FRACTION || true
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

command=(
    python -u "$TOPAX_ROOT/visualization/wang_2025/r800.py"
    --output-dir "$VISUALIZATION_OUTPUT_DIR"
)

cd "$TOPAX_ROOT"
"${command[@]}"

