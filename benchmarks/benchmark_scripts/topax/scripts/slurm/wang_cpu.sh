#!/usr/bin/env bash
set -euo pipefail

# Submit one Topax Wang CPU configuration. A single execution provides both
# the design-iteration runtime and the process peak resident memory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
# shellcheck disable=SC1091
source "$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/scripts/common.sh"

RESOLUTIONS=(96 192 384 768 800)
MESH_X=(96 192 384 768 800)
MESH_Y=(48 96 192 384 400)
MESH_Z=(48 96 192 384 400)

usage() {
    cat <<'EOF'
Usage: wang_cpu.sh --levels N --crossover-level N [options]

Options:
  --run-id ID              UTC ID in YYYYMMDDTHHMMSSZ format (default: now)
  --levels N               total grid levels, including the fine grid
  --crossover-level N      first materialized Galerkin level
EOF
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    RUN_ID=""
    LEVELS=""
    CROSSOVER=""
    while (( $# )); do
        case "$1" in
            --run-id) RUN_ID="${2:?}"; shift 2 ;;
            --levels) LEVELS="${2:?}"; shift 2 ;;
            --crossover-level) CROSSOVER="${2:?}"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) topax_die "unknown argument: $1" ;;
        esac
    done
    [[ -n "$LEVELS" ]] || topax_die "--levels is required"
    [[ -n "$CROSSOVER" ]] || topax_die "--crossover-level is required"
    topax_validate_hierarchy "$LEVELS" "$CROSSOVER"

    ACCOUNT="${ACCOUNT:-naiss2026-4-1571-cpu}"
    PARTITION="${PARTITION:-cpu}"
    CPUS="${CPUS:-32}"
    WALLTIME="${WALLTIME:-12:00:00}"
    LOG_ROOT="$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/scripts/.slurm"
    RUN_ID="$(topax_prepare_run_id "$RUN_ID" "$LOG_ROOT")"
    LOG_DIR="$LOG_ROOT/$RUN_ID"

    job_id="$(
        sbatch --parsable \
            -A "$ACCOUNT" -p "$PARTITION" -N 1 --ntasks=1 \
            --cpus-per-task="$CPUS" -t "$WALLTIME" \
            --array=0-4 -J topax-wang-cpu \
            -o "$LOG_DIR/wang-cpu-%A_%a.out" \
            --export=ALL,TOPAX_RUN_ID="$RUN_ID",TOPAX_LEVELS="$LEVELS",TOPAX_CROSSOVER="$CROSSOVER",TOPAX_ROOT="$TOPAX_ROOT" \
            "$0"
    )"
    echo "Run ID: $RUN_ID"
    echo "Submitted CPU array: $job_id"
    exit 0
fi

RUN_ID="${TOPAX_RUN_ID:?}"
LEVELS="${TOPAX_LEVELS:?}"
CROSSOVER="${TOPAX_CROSSOVER:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"
RESOLUTION="${RESOLUTIONS[$INDEX]}"
EX="${MESH_X[$INDEX]}"
EY="${MESH_Y[$INDEX]}"
EZ="${MESH_Z[$INDEX]}"
MESH="${EX}x${EY}x${EZ}"
OUTPUT_DIR="$(topax_result_dir "$TOPAX_ROOT" wang_2025 cpu "$MESH" "$RUN_ID")"
LOCK_RUNTIME="$(topax_acquire_result_lock "$OUTPUT_DIR" runtime)"
LOCK_MEMORY=""
SCRATCH="${SLURM_TMPDIR:-/tmp}/topax-wang-cpu-${SLURM_JOB_ID}-${INDEX}"

cleanup() {
    [[ -z "$LOCK_RUNTIME" ]] || rm -rf "$LOCK_RUNTIME"
    [[ -z "$LOCK_MEMORY" ]] || rm -rf "$LOCK_MEMORY"
    rm -f "$OUTPUT_DIR/.runtime.$$.csv" "$OUTPUT_DIR/.memory.$$.csv"
    rm -rf "$SCRATCH"
}
trap cleanup EXIT INT TERM

LOCK_MEMORY="$(topax_acquire_result_lock "$OUTPUT_DIR" memory)"
mkdir -p "$SCRATCH"

CPU_ENV="${TOPAX_CPU_ENV:-$HOME/envs/topax-cpu.sh}"
[[ -f "$CPU_ENV" ]] || topax_die "CPU environment script not found"
# shellcheck disable=SC1090
source "$CPU_ENV"
export LC_ALL=C

RAW="$SCRATCH/output.txt"
TIME_FILE="$SCRATCH/time.txt"
ITERATION_FILE="$SCRATCH/runtime.csv"
COMMAND=(
    python -u "$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py"
    --resolution "$RESOLUTION"
    --levels "$LEVELS"
    --galerkin-crossover-level "$CROSSOVER"
    --smoothing-steps 1
    --max-iterations 50
    --runtime-csv "$ITERATION_FILE"
)

set +e
/usr/bin/time -v -o "$TIME_FILE" "${COMMAND[@]}" > "$RAW" 2>&1
STATUS=$?
set -e
if (( STATUS != 0 )); then
    cat "$RAW" >&2
    exit "$STATUS"
fi

TEMP_RUNTIME="$OUTPUT_DIR/.runtime.$$.csv"
TEMP_MEMORY="$OUTPUT_DIR/.memory.$$.csv"
topax_write_memory_csv "$TEMP_MEMORY" "$ITERATION_FILE" "$TIME_FILE"
mv "$ITERATION_FILE" "$TEMP_RUNTIME"

topax_write_configuration \
    "$OUTPUT_DIR" wang_2025 cpu "$EX" "$EY" "$EZ" \
    "$RESOLUTION" "$LEVELS" "$CROSSOVER" 1 50 50 1e-3 1e-4 \
    true true false
mv "$TEMP_RUNTIME" "$OUTPUT_DIR/runtime.csv"
mv "$TEMP_MEMORY" "$OUTPUT_DIR/memory.csv"
echo "Wrote $OUTPUT_DIR/{runtime,memory}.csv"
