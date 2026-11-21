#!/usr/bin/env bash
set -euo pipefail

# Submit one Topax Wang GPU configuration over the common exact-dyadic mesh
# series. Runtime and memory measurements share one run ID.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
# shellcheck disable=SC1091
source "$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/scripts/common.sh"

RESOLUTIONS=(96 192 384 768 800 1024)
MESH_X=(96 192 384 768 800 1024)
MESH_Y=(48 96 192 384 400 512)
MESH_Z=(48 96 192 384 400 512)

usage() {
    cat <<'EOF'
Usage: wang_gpu.sh --levels N --crossover-level N [options]

Options:
  --run-id ID              UTC ID in YYYYMMDDTHHMMSSZ format (default: now)
  --measurement KIND       runtime, memory, or both (default: both)
  --task-index N           submit only mesh index N, from 0 through 5
  --max-resolution R       largest submitted resolution (default: 1024)
  --levels N               total grid levels, including the fine grid
  --crossover-level N      first materialized Galerkin level
EOF
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    RUN_ID=""
    LEVELS=""
    CROSSOVER=""
    MEASUREMENT="both"
    TASK_INDEX=""
    MAX_RESOLUTION=1024
    while (( $# )); do
        case "$1" in
            --run-id) RUN_ID="${2:?}"; shift 2 ;;
            --measurement) MEASUREMENT="${2:?}"; shift 2 ;;
            --task-index) TASK_INDEX="${2:?}"; shift 2 ;;
            --max-resolution) MAX_RESOLUTION="${2:?}"; shift 2 ;;
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
    if [[ -n "$TASK_INDEX" ]]; then
        [[ "$TASK_INDEX" =~ ^[0-5]$ ]] ||
            topax_die "--task-index must be an integer from 0 through 5"
    fi
    case "$MAX_RESOLUTION" in
        96) MAX_INDEX=0 ;;
        192) MAX_INDEX=1 ;;
        384) MAX_INDEX=2 ;;
        768) MAX_INDEX=3 ;;
        800) MAX_INDEX=4 ;;
        1024) MAX_INDEX=5 ;;
        *) topax_die "--max-resolution must be one of: 96, 192, 384, 768, 800, 1024" ;;
    esac
    if [[ -n "$TASK_INDEX" ]] && (( TASK_INDEX > MAX_INDEX )); then
        topax_die "--task-index exceeds --max-resolution"
    fi

    ACCOUNT="${ACCOUNT:-naiss2026-4-1571-gpu}"
    PARTITION="${PARTITION:-gpu}"
    CPUS="${CPUS:-72}"
    WALLTIME="${WALLTIME:-04:00:00}"
    LOG_ROOT="$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/scripts/.slurm"
    RUN_ID="$(topax_prepare_run_id "$RUN_ID" "$LOG_ROOT")"
    LOG_DIR="$LOG_ROOT/$RUN_ID"

    measurements=("$MEASUREMENT")
    [[ "$MEASUREMENT" != both ]] || measurements=(runtime memory)
    ARRAY_SPEC="${TASK_INDEX:-0-$MAX_INDEX}"
    echo "Run ID: $RUN_ID"
    for kind in "${measurements[@]}"; do
        job_id="$(
            sbatch --parsable \
                -A "$ACCOUNT" -p "$PARTITION" -N 1 --ntasks=1 \
                --cpus-per-task="$CPUS" --gpus=1 -t "$WALLTIME" \
                --array="$ARRAY_SPEC" -J "topax-wang-gpu-${kind}" \
                -o "$LOG_DIR/wang-gpu-${kind}-%A_%a.out" \
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
RESOLUTION="${RESOLUTIONS[$INDEX]}"
EX="${MESH_X[$INDEX]}"
EY="${MESH_Y[$INDEX]}"
EZ="${MESH_Z[$INDEX]}"
MESH="${EX}x${EY}x${EZ}"
OUTPUT_DIR="$(topax_result_dir "$TOPAX_ROOT" wang_2025 gpu "$MESH" "$RUN_ID")"
LOCK="$(topax_acquire_result_lock "$OUTPUT_DIR" "$MEASUREMENT")"
SCRATCH="${SLURM_TMPDIR:-/tmp}/topax-wang-gpu-${SLURM_JOB_ID}-${INDEX}-${MEASUREMENT}"
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

RUNTIME_JAX_PREALLOCATE=true
[[ "$RESOLUTION" != 1024 ]] || RUNTIME_JAX_PREALLOCATE=false

if [[ "$MEASUREMENT" == memory || "$RUNTIME_JAX_PREALLOCATE" == false ]]; then
    unset TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_MEM_FRACTION XLA_CLIENT_MEM_FRACTION || true
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=address
    if [[ "$MEASUREMENT" == memory ]]; then
        MAX_ITERATIONS=10
    else
        MAX_ITERATIONS=50
    fi
else
    unset XLA_PYTHON_CLIENT_ALLOCATOR XLA_CLIENT_MEM_FRACTION || true
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
    MAX_ITERATIONS=50
fi

RAW="$SCRATCH/output.txt"
TIME_FILE="$SCRATCH/time.txt"
GPU_FILE="$SCRATCH/gpu.csv"
ITERATION_FILE="$SCRATCH/runtime.csv"
if [[ "$MEASUREMENT" == memory ]]; then
    topax_start_gpu_sampler "$GPU_FILE" "${GPU_SAMPLE_INTERVAL:-0.2}"
fi

COMMAND=(
    python -u "$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py"
    --resolution "$RESOLUTION"
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
    "$OUTPUT_DIR" wang_2025 gpu "$EX" "$EY" "$EZ" \
    "$RESOLUTION" "$LEVELS" "$CROSSOVER" 1 50 10 1e-3 1e-4 \
    true false "$RUNTIME_JAX_PREALLOCATE"
mv "$TEMP_RESULT" "$OUTPUT_DIR/$MEASUREMENT.csv"
echo "Wrote $OUTPUT_DIR/$MEASUREMENT.csv"
