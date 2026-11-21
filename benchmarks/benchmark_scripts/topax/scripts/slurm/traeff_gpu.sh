#!/usr/bin/env bash
set -euo pipefail

# Submit one Topax Träff GPU configuration. Runtime and memory are measured in
# separate Slurm arrays but share the same run ID and result directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
# shellcheck disable=SC1091
source "$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/scripts/common.sh"

MESH_X=(64 80 96 128 160 192 256 320 400 512 640)
MESH_Y=(32 40 48 64 80 96 128 160 200 256 320)
MESH_Z=(32 40 48 64 80 96 128 160 200 256 320)

usage() {
    cat <<'EOF'
Usage: traeff_gpu.sh --levels N --crossover-level N [options]

Options:
  --run-id ID              UTC ID in YYYYMMDDTHHMMSSZ format (default: now)
  --measurement KIND       runtime, memory, or both (default: both)
  --levels N               total grid levels, including the fine grid
  --crossover-level N      first materialized Galerkin level
EOF
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    RUN_ID=""
    LEVELS=""
    CROSSOVER=""
    MEASUREMENT="both"
    while (( $# )); do
        case "$1" in
            --run-id) RUN_ID="${2:?}"; shift 2 ;;
            --measurement) MEASUREMENT="${2:?}"; shift 2 ;;
            --levels) LEVELS="${2:?}"; shift 2 ;;
            --crossover-level) CROSSOVER="${2:?}"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) topax_die "unknown argument: $1" ;;
        esac
    done
    [[ -n "$LEVELS" ]] || topax_die "--levels is required"
    [[ -n "$CROSSOVER" ]] || topax_die "--crossover-level is required"
    topax_validate_hierarchy "$LEVELS" "$CROSSOVER"
    topax_validate_measurement "$MEASUREMENT"

    ACCOUNT="${ACCOUNT:-naiss2026-4-1571-gpu}"
    PARTITION="${PARTITION:-gpu}"
    CPUS="${CPUS:-72}"
    WALLTIME="${WALLTIME:-02:00:00}"
    LOG_ROOT="$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/scripts/.slurm"
    RUN_ID="$(topax_prepare_run_id "$RUN_ID" "$LOG_ROOT")"
    LOG_DIR="$LOG_ROOT/$RUN_ID"

    measurements=("$MEASUREMENT")
    [[ "$MEASUREMENT" != both ]] || measurements=(runtime memory)
    echo "Run ID: $RUN_ID"
    for kind in "${measurements[@]}"; do
        job_id="$(
            sbatch --parsable \
                -A "$ACCOUNT" -p "$PARTITION" -N 1 --ntasks=1 \
                --cpus-per-task="$CPUS" --gpus=1 -t "$WALLTIME" \
                --array=0-10 -J "topax-traeff-${kind}" \
                -o "$LOG_DIR/traeff-${kind}-%A_%a.out" \
                --export=ALL,TOPAX_RUN_ID="$RUN_ID",TOPAX_LEVELS="$LEVELS",TOPAX_CROSSOVER="$CROSSOVER",TOPAX_MEASUREMENT="$kind",TOPAX_ROOT="$TOPAX_ROOT" \
                "$0"
        )"
        echo "Submitted $kind array: $job_id"
    done
    exit 0
fi

RUN_ID="${TOPAX_RUN_ID:?}"
LEVELS="${TOPAX_LEVELS:?}"
CROSSOVER="${TOPAX_CROSSOVER:?}"
MEASUREMENT="${TOPAX_MEASUREMENT:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"
EX="${MESH_X[$INDEX]}"
EY="${MESH_Y[$INDEX]}"
EZ="${MESH_Z[$INDEX]}"
MESH="${EX}x${EY}x${EZ}"
OUTPUT_DIR="$(topax_result_dir "$TOPAX_ROOT" traeff_2023 gpu "$MESH" "$RUN_ID")"
LOCK="$(topax_acquire_result_lock "$OUTPUT_DIR" "$MEASUREMENT")"
SCRATCH="${SLURM_TMPDIR:-/tmp}/topax-traeff-${SLURM_JOB_ID}-${INDEX}-${MEASUREMENT}"
mkdir -p "$SCRATCH"

cleanup() {
    topax_stop_gpu_sampler
    rm -f "$OUTPUT_DIR/.${MEASUREMENT}.$$.csv"
    rm -rf "$LOCK" "$SCRATCH"
}
trap cleanup EXIT INT TERM

GPU_ENV="${TOPAX_GPU_ENV:-$HOME/envs/topax-gpu.sh}"
[[ -f "$GPU_ENV" ]] || topax_die "GPU environment script not found"
# shellcheck disable=SC1090
source "$GPU_ENV"
export LC_ALL=C

if [[ "$MEASUREMENT" == memory ]]; then
    unset TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_MEM_FRACTION XLA_CLIENT_MEM_FRACTION || true
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=address
    MAX_ITERATIONS=10
else
    unset XLA_PYTHON_CLIENT_ALLOCATOR XLA_CLIENT_MEM_FRACTION || true
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
    MAX_ITERATIONS=20
fi

RAW="$SCRATCH/output.txt"
TIME_FILE="$SCRATCH/time.txt"
GPU_FILE="$SCRATCH/gpu.csv"
ITERATION_FILE="$SCRATCH/runtime.csv"
if [[ "$MEASUREMENT" == memory ]]; then
    topax_start_gpu_sampler "$GPU_FILE" "${GPU_SAMPLE_INTERVAL:-0.2}"
fi

COMMAND=(
    python -u "$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py"
    --elements "$EX" "$EY" "$EZ"
    --levels "$LEVELS"
    --galerkin-crossover-level "$CROSSOVER"
    --smoothing-steps 1
    --max-iterations "$MAX_ITERATIONS"
    --runtime-csv "$ITERATION_FILE"
)

set +e
/usr/bin/time -v -o "$TIME_FILE" "${COMMAND[@]}" > "$RAW" 2>&1
STATUS=$?
set -e
topax_stop_gpu_sampler

if (( STATUS != 0 )); then
    cat "$RAW" >&2
    exit "$STATUS"
fi

TEMP_RESULT="$OUTPUT_DIR/.${MEASUREMENT}.$$.csv"
if [[ "$MEASUREMENT" == runtime ]]; then
    mv "$ITERATION_FILE" "$TEMP_RESULT"
else
    topax_write_memory_csv \
        "$TEMP_RESULT" "$ITERATION_FILE" "$TIME_FILE" \
        "$GPU_FILE" "${TOPAX_GPU_BASELINE_MIB:-}"
fi

topax_write_configuration \
    "$OUTPUT_DIR" traeff_2023 gpu "$EX" "$EY" "$EZ" \
    not_applicable "$LEVELS" "$CROSSOVER" 1 20 10 1e-5 '"not_applicable"' \
    true false true
mv "$TEMP_RESULT" "$OUTPUT_DIR/$MEASUREMENT.csv"
echo "Wrote $OUTPUT_DIR/$MEASUREMENT.csv"
