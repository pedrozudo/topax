#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFANTOM_BENCH_DIR="${PYFANTOM_BENCH_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$PYFANTOM_BENCH_DIR/../../.." && pwd)}"
PYFANTOM_SOURCE="$PYFANTOM_BENCH_DIR/pyFANTOM"
PYTHON_BIN="$PYFANTOM_BENCH_DIR/.venv/bin/python"
SOURCE_REVISION="ffa4c95f087f86933629cb1926030b80501c3136"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

if [[ -n "${CUDA_HOME:-}" ]]; then
    [[ -d "$CUDA_HOME" ]] || die "CUDA_HOME does not exist: $CUDA_HOME"
    CUDA_LIBDIR=""
    for candidate in \
        "$CUDA_HOME/targets/sbsa-linux/lib" \
        "$CUDA_HOME/targets/x86_64-linux/lib" \
        "$CUDA_HOME/lib64"; do
        if [[ -e "$candidate/libnvrtc.so.13" ]]; then
            CUDA_LIBDIR="$candidate"
            break
        fi
    done
    [[ -n "$CUDA_LIBDIR" ]] || die "libnvrtc.so.13 was not found below CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -n "${CHOLMOD_ROOT:-}" ]]; then
    [[ -d "$CHOLMOD_ROOT" ]] || die "CHOLMOD_ROOT does not exist: $CHOLMOD_ROOT"
    if [[ -d "$CHOLMOD_ROOT/lib64" ]]; then
        CHOLMOD_LIBDIR="$CHOLMOD_ROOT/lib64"
    elif [[ -d "$CHOLMOD_ROOT/lib" ]]; then
        CHOLMOD_LIBDIR="$CHOLMOD_ROOT/lib"
    else
        die "CHOLMOD_ROOT contains neither lib64 nor lib"
    fi
    export LD_LIBRARY_PATH="$CHOLMOD_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

MESH_X=(64 80 96 128 160 192 256 320 400 512 640)
MESH_Y=(32 40 48 64 80 96 128 160 200 256 320)
MESH_Z=(32 40 48 64 80 96 128 160 200 256 320)

usage() {
    cat <<'EOF'
Usage: traeff_gpu.sh [options]

Options:
  --run-id ID          UTC ID in YYYYMMDDTHHMMSSZ format (default: now)
  --measurement KIND   runtime, memory, or both (default: both)
  --task-index N       submit only mesh index N, from 0 through 10
EOF
}

validate_measurement() {
    case "$1" in
        runtime|memory|both) ;;
        *) die "measurement must be runtime, memory, or both" ;;
    esac
}

prepare_run_id() {
    local requested="$1"
    local log_root="$2"
    local candidate

    mkdir -p "$log_root"
    if [[ -n "$requested" ]]; then
        [[ "$requested" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] ||
            die "run ID must use the UTC format YYYYMMDDTHHMMSSZ"
        mkdir -p "$log_root/$requested"
        printf '%s\n' "$requested"
        return
    fi

    while true; do
        candidate="$(date -u +%Y%m%dT%H%M%SZ)"
        if mkdir "$log_root/$candidate" 2>/dev/null; then
            printf '%s\n' "$candidate"
            return
        fi
        sleep 1
    done
}

acquire_result_lock() {
    local output_dir="$1"
    local measurement="$2"
    local result="$output_dir/$measurement.csv"
    local lock="$output_dir/.$measurement.lock"

    mkdir -p "$output_dir"
    [[ ! -e "$result" ]] || die "result already exists: $result"
    mkdir "$lock" 2>/dev/null ||
        die "another $measurement measurement is using $output_dir"
    printf '%s\n' "$lock"
}

gpu_used_mib() {
    local device="${CUDA_VISIBLE_DEVICES:-0}"
    device="${device%%,*}"
    nvidia-smi -i "$device" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null |
        head -n 1 | tr -d '[:space:]'
}

start_gpu_sampler() {
    local output="$1"
    local interval="$2"

    GPU_BASELINE_MIB="$(gpu_used_mib || true)"
    [[ "$GPU_BASELINE_MIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || GPU_BASELINE_MIB=""
    echo "timestamp_epoch_s,used_mib" > "$output"
    (
        while true; do
            local used
            used="$(gpu_used_mib || true)"
            if [[ "$used" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                printf '%s,%s\n' "$(date +%s.%N)" "$used" >> "$output"
            fi
            sleep "$interval"
        done
    ) &
    GPU_SAMPLER_PID=$!
}

stop_gpu_sampler() {
    if [[ -n "${GPU_SAMPLER_PID:-}" ]]; then
        kill "$GPU_SAMPLER_PID" 2>/dev/null || true
        wait "$GPU_SAMPLER_PID" 2>/dev/null || true
        unset GPU_SAMPLER_PID
    fi
}

write_memory_csv() {
    local output="$1"
    local runtime_csv="$2"
    local time_file="$3"
    local gpu_file="$4"
    local baseline_mib="$5"
    local iterations peak_rss_kib host_peak_gib peak_mib

    iterations="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$runtime_csv")"
    (( iterations > 0 )) || die "runtime CSV contains no iterations"
    peak_rss_kib="$(
        awk -F: '/Maximum resident set size \(kbytes\)/ {
            gsub(/[[:space:]]/, "", $2); print $2; exit
        }' "$time_file"
    )"
    [[ "$peak_rss_kib" =~ ^[0-9]+$ ]] || die "could not read peak resident memory"
    [[ "$baseline_mib" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "GPU baseline is unavailable"
    peak_mib="$(
        awk -F, 'NR > 1 && $2 ~ /^[0-9]+([.][0-9]+)?$/ {
            if (!seen || $2 > peak) peak = $2; seen = 1
        } END { if (seen) print peak }' "$gpu_file"
    )"
    [[ "$peak_mib" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "GPU samples are unavailable"
    host_peak_gib="$(awk -v value="$peak_rss_kib" 'BEGIN {printf "%.12g", value / 1048576}')"

    {
        echo "design_iterations_completed,host_peak_rss_gib,gpu_baseline_gib,gpu_peak_gib,gpu_peak_above_baseline_gib"
        awk -v iterations="$iterations" -v host="$host_peak_gib" \
            -v baseline="$baseline_mib" -v peak="$peak_mib" \
            'BEGIN {printf "%s,%s,%.12g,%.12g,%.12g\n", iterations, host, baseline/1024, peak/1024, (peak-baseline)/1024}'
    } > "$output"
}

write_configuration() {
    local output_dir="$1"
    local temporary="$output_dir/.configuration.$$.toml"
    local destination="$output_dir/configuration.toml"
    local python_version numpy_version scipy_version cupy_version sksparse_version
    local sampling_interval="${GPU_SAMPLE_INTERVAL:-0.2}"

    python_version="$("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')"
    numpy_version="$("$PYTHON_BIN" -c 'import numpy; print(numpy.__version__)')"
    scipy_version="$("$PYTHON_BIN" -c 'import scipy; print(scipy.__version__)')"
    cupy_version="$("$PYTHON_BIN" -c 'import cupy; print(cupy.__version__)')"
    sksparse_version="$("$PYTHON_BIN" -c 'import importlib.metadata; print(importlib.metadata.version("scikit-sparse"))')"

    cat > "$temporary" <<EOF
schema_version = 2
benchmark = "traeff_2023"
implementation = "pyfantom"
implementation_version = "0.1.0"
source_revision = "$SOURCE_REVISION"
instrumentation_patch = "record_pcg_state.patch"
backend = "gpu"
elements = [$EX, $EY, $EZ]
levels = 4
coarsenings = 3
cycle = "W"
w_cycle_level = 1
smoothing_steps = 3
jacobi_damping = 0.5
minimum_jacobi_damping = 0.4
coarse_solver = "CHOLMOD"
precision = "float64"
physics_tolerance = 1e-5
physics_max_iterations = 1000
filter = "linear density filter"
filter_radius_elements = 1.5
volume_fraction = 0.1
simp_penalty = 3
minimum_relative_stiffness = 1e-6
python_version = "$python_version"
numpy_version = "$numpy_version"
scipy_version = "$scipy_version"
cupy_version = "$cupy_version"
scikit_sparse_version = "$sksparse_version"

[measurements.runtime]
design_iteration_limit = 20
full_benchmark = true
cupy_allocator = "default_memory_pool"
explicit_memory_preallocation = false
release_unused_pool_blocks_after_linear_solve = true

[measurements.memory]
design_iteration_limit = 10
full_benchmark = false
cupy_allocator = "default_memory_pool"
explicit_memory_preallocation = false
release_unused_pool_blocks_after_linear_solve = true
sampling_interval_seconds = $sampling_interval
EOF

    if [[ -e "$destination" ]]; then
        cmp -s "$temporary" "$destination" || {
            rm -f "$temporary"
            die "configuration mismatch in $output_dir"
        }
        rm -f "$temporary"
    else
        mv "$temporary" "$destination"
    fi
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    RUN_ID=""
    MEASUREMENT="both"
    TASK_INDEX=""
    while (( $# )); do
        case "$1" in
            --run-id) RUN_ID="${2:?}"; shift 2 ;;
            --measurement) MEASUREMENT="${2:?}"; shift 2 ;;
            --task-index) TASK_INDEX="${2:?}"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    validate_measurement "$MEASUREMENT"
    if [[ -n "$TASK_INDEX" ]]; then
        [[ "$TASK_INDEX" =~ ^([0-9]|10)$ ]] ||
            die "--task-index must be an integer from 0 through 10"
    fi
    [[ -x "$PYTHON_BIN" ]] || die "run setup.sh before submitting benchmarks"
    [[ -d "$PYFANTOM_SOURCE/.git" ]] || die "pyFANTOM checkout is missing"
    [[ "$(git -C "$PYFANTOM_SOURCE" rev-parse HEAD)" == "$SOURCE_REVISION" ]] ||
        die "pyFANTOM checkout is not at the pinned revision"
    git -C "$PYFANTOM_SOURCE" apply --reverse --check \
        "$PYFANTOM_BENCH_DIR/patches/record_pcg_state.patch" >/dev/null 2>&1 ||
        die "pyFANTOM PCG instrumentation patch is missing"

    SUBMISSION_ACCOUNT="${ACCOUNT:-${SLURM_JOB_ACCOUNT:-}}"
    ACCOUNT_ARGUMENT=()
    [[ -z "$SUBMISSION_ACCOUNT" ]] || ACCOUNT_ARGUMENT=(-A "$SUBMISSION_ACCOUNT")
    PARTITION="${PARTITION:-gpu}"
    CPUS="${CPUS:-72}"
    WALLTIME="${WALLTIME:-04:00:00}"
    ARRAY_SPEC="${TASK_INDEX:-0-10}"
    LOG_ROOT="$PYFANTOM_BENCH_DIR/.slurm"
    RUN_ID="$(prepare_run_id "$RUN_ID" "$LOG_ROOT")"
    LOG_DIR="$LOG_ROOT/$RUN_ID"
    measurements=("$MEASUREMENT")
    [[ "$MEASUREMENT" != both ]] || measurements=(runtime memory)

    echo "Run ID: $RUN_ID"
    for kind in "${measurements[@]}"; do
        job_id="$(
            sbatch --parsable \
                "${ACCOUNT_ARGUMENT[@]}" -p "$PARTITION" -N 1 --ntasks=1 \
                --cpus-per-task="$CPUS" --gpus=1 -t "$WALLTIME" \
                --array="$ARRAY_SPEC" -J "pyfantom-traeff-${kind}" \
                -o "$LOG_DIR/traeff-${kind}-%A_%a.out" \
                --export=ALL,PYFANTOM_RUN_ID="$RUN_ID",PYFANTOM_MEASUREMENT="$kind",PYFANTOM_BENCH_DIR="$PYFANTOM_BENCH_DIR",TOPAX_ROOT="$TOPAX_ROOT" \
                "$0"
        )"
        echo "Submitted $kind array: $job_id"
    done
    exit 0
fi

RUN_ID="${PYFANTOM_RUN_ID:?}"
MEASUREMENT="${PYFANTOM_MEASUREMENT:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"
EX="${MESH_X[$INDEX]}"
EY="${MESH_Y[$INDEX]}"
EZ="${MESH_Z[$INDEX]}"
MESH="${EX}x${EY}x${EZ}"
OUTPUT_DIR="$TOPAX_ROOT/benchmarks/results/traeff_2023/pyfantom/gpu/$MESH/$RUN_ID"
LOCK="$(acquire_result_lock "$OUTPUT_DIR" "$MEASUREMENT")"
SCRATCH="${SLURM_TMPDIR:-/tmp}/pyfantom-traeff-${SLURM_JOB_ID}-${INDEX}-${MEASUREMENT}"
mkdir -p "$SCRATCH"

cleanup() {
    stop_gpu_sampler
    rm -f "$OUTPUT_DIR/.${MEASUREMENT}.$$.csv" "$OUTPUT_DIR/.configuration.$$.toml"
    rm -rf "$LOCK" "$SCRATCH"
}
trap cleanup EXIT INT TERM

[[ -x "$PYTHON_BIN" ]] || die "pyFANTOM virtual environment is missing"
"$PYTHON_BIN" -c 'import ctypes; ctypes.CDLL("libnvrtc.so.13")' >/dev/null 2>&1 ||
    die "libnvrtc.so.13 is unavailable; set CUDA_HOME or load the CUDA 13 runtime environment"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-72}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
export LC_ALL=C

if [[ "$MEASUREMENT" == memory ]]; then
    MAX_ITERATIONS=10
else
    MAX_ITERATIONS=20
fi

RAW="$SCRATCH/output.txt"
TIME_FILE="$SCRATCH/time.txt"
GPU_FILE="$SCRATCH/gpu.csv"
ITERATION_FILE="$SCRATCH/runtime.csv"
if [[ "$MEASUREMENT" == memory ]]; then
    start_gpu_sampler "$GPU_FILE" "${GPU_SAMPLE_INTERVAL:-0.2}"
fi

COMMAND=(
    "$PYTHON_BIN" -u "$PYFANTOM_BENCH_DIR/traeff_2023/cantilever.py"
    --elements "$EX" "$EY" "$EZ"
    --max-iterations "$MAX_ITERATIONS"
    --runtime-csv "$ITERATION_FILE"
)

set +e
/usr/bin/time -v -o "$TIME_FILE" "${COMMAND[@]}" > "$RAW" 2>&1
STATUS=$?
set -e
stop_gpu_sampler
if (( STATUS != 0 )); then
    cat "$RAW" >&2
    exit "$STATUS"
fi

TEMP_RESULT="$OUTPUT_DIR/.${MEASUREMENT}.$$.csv"
if [[ "$MEASUREMENT" == runtime ]]; then
    mv "$ITERATION_FILE" "$TEMP_RESULT"
else
    write_memory_csv \
        "$TEMP_RESULT" "$ITERATION_FILE" "$TIME_FILE" \
        "$GPU_FILE" "${GPU_BASELINE_MIB:-}"
fi

write_configuration "$OUTPUT_DIR"
mv "$TEMP_RESULT" "$OUTPUT_DIR/$MEASUREMENT.csv"
echo "Wrote $OUTPUT_DIR/$MEASUREMENT.csv"
