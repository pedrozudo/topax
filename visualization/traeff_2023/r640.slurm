#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    : "${ACCOUNT:?set ACCOUNT to the GPU allocation account}"
    : "${VISUALIZATION_OUTPUT_DIR:?set VISUALIZATION_OUTPUT_DIR to a new output directory}"

    PARTITION="${PARTITION:-gpu}"
    CPUS="${CPUS:-72}"
    WALLTIME="${WALLTIME:-01:00:00}"
    MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
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
            -J topax-traeff-r640-visualization \
            -o "$LOG_DIR/traeff-r640-%j.out" \
            --export=ALL,TOPAX_ROOT="$TOPAX_ROOT",VISUALIZATION_OUTPUT_DIR="$VISUALIZATION_OUTPUT_DIR",MAX_ITERATIONS="$MAX_ITERATIONS",TOPAX_GPU_ENV="$GPU_ENV" \
            "$0"
    )"
    echo "Submitted Träff R640 visualization job: $job_id"
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

cd "$TOPAX_ROOT"
python -u visualization/traeff_2023/r640.py \
    --output-dir "$VISUALIZATION_OUTPUT_DIR" \
    --max-iterations "${MAX_ITERATIONS:-50}"
