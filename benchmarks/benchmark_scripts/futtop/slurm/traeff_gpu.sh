#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
FUTTOP_BENCH_DIR="${FUTTOP_BENCH_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$FUTTOP_BENCH_DIR/../../.." && pwd)}"
FUTTOP_SOURCE="$FUTTOP_BENCH_DIR/Futtop"
FUTTOP_BINARY="$FUTTOP_BENCH_DIR/build/futtop"
SOURCE_REVISION="c3cd2b00632ad57cd3d8917246552b2fb4e99e01"

MESH_X=(64 80 96 128 160 192 256 320 400 512 640)
MESH_Y=(32 40 48 64 80 96 128 160 200 256 320)
MESH_Z=(32 40 48 64 80 96 128 160 200 256 320)

die() {
    echo "ERROR: $*" >&2
    exit 2
}

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

parse_runtime() {
    local raw="$1"
    local output="$2"

    python3 - "$raw" "$output" <<'PY'
import csv
import re
import sys
from pathlib import Path

raw_path, output_path = map(Path, sys.argv[1:])
text = raw_path.read_text(errors="replace")
number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
iteration_pattern = re.compile(
    rf"^It\.:\s*(\d+)\s+Obj\.:\s*({number})\s+Vol\.:\s*({number})\s+"
    rf"ch\.:\s*({number})\s+relres:\s*({number})\s+iters:\s*(\d+)\s*$",
    re.MULTILINE,
)
time_pattern = re.compile(
    rf"^\s*assembly:\s*({number})s\s+solve:\s*({number})s\s+"
    rf"update:\s*({number})s\s+total:\s*({number})s\s*$",
    re.MULTILINE,
)
iterations = list(iteration_pattern.finditer(text))
times = list(time_pattern.finditer(text))
if not iterations:
    raise SystemExit("Futtop output contains no completed design iterations")
if len(iterations) != len(times):
    raise SystemExit(
        f"found {len(iterations)} iteration records but {len(times)} timing records"
    )

fields = [
    "iteration",
    "compliance",
    "mean_design_density",
    "mean_physical_density",
    "volume_constraint",
    "non_discreteness_reported",
    "change",
    "physics_cg_iterations",
    "physics_relative_residual",
    "filter_cg_iterations",
    "filter_relative_residual",
    "iteration_time_s",
]
with output_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for iteration, timing in zip(iterations, times):
        volume = float(iteration.group(3))
        writer.writerow(
            {
                "iteration": int(iteration.group(1)),
                "compliance": float(iteration.group(2)),
                "mean_design_density": "",
                "mean_physical_density": volume,
                "volume_constraint": volume - 0.1,
                "non_discreteness_reported": "",
                "change": float(iteration.group(4)),
                "physics_cg_iterations": int(iteration.group(6)),
                "physics_relative_residual": float(iteration.group(5)),
                "filter_cg_iterations": "",
                "filter_relative_residual": "",
                "iteration_time_s": float(timing.group(4)),
            }
        )
PY
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

toml_string() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

write_configuration() {
    local output_dir="$1"
    local temporary="$output_dir/.configuration.$$.toml"
    local destination="$output_dir/configuration.toml"
    local futhark_version futhark_revision compiler_version sampling_interval

    futhark_version="$(head -n 1 "$FUTTOP_BENCH_DIR/build/futhark-version.txt")"
    futhark_revision="$(sed -n 's/^git:[[:space:]]*//p' "$FUTTOP_BENCH_DIR/build/futhark-version.txt")"
    compiler_version="$(head -n 1 "$FUTTOP_BENCH_DIR/build/compiler-version.txt")"
    futhark_version="$(toml_string "$futhark_version")"
    futhark_revision="$(toml_string "$futhark_revision")"
    compiler_version="$(toml_string "$compiler_version")"
    sampling_interval="${GPU_SAMPLE_INTERVAL:-0.2}"

    cat > "$temporary" <<EOF
schema_version = 2
benchmark = "traeff_2023"
implementation = "futtop"
implementation_version = "0.9.1"
source_revision = "$SOURCE_REVISION"
output_patch = "disable_vtr_output.patch"
backend = "opencl_gpu"
elements = [$EX, $EY, $EZ]
levels = 4
coarsenings = 3
cycle = "V"
first_coarsened_grid_active = false
smoother = "nodal symmetric successive over-relaxation"
sor_damping = 0.6
pre_smoothing_steps = 1
post_smoothing_steps = 1
coarse_solver = "fixed-iteration Jacobi-preconditioned conjugate gradients"
coarse_solver_iterations = 800
precision = "mixed_float32_float64"
physics_tolerance = 1e-5
physics_max_iterations = 200
filter = "linear density filter"
filter_radius_elements = 1.5
volume_fraction = 0.1
simp_penalty = 3
minimum_relative_stiffness = 1e-6
optimality_criteria_move_limit = 0.1
futhark_version = "$futhark_version"
futhark_revision = "$futhark_revision"
c_compiler = "$compiler_version"

[measurements.runtime]
design_iteration_limit = 20
full_benchmark = true
device_allocator = "Futhark OpenCL runtime"
explicit_memory_preallocation = false

[measurements.memory]
design_iteration_limit = 10
full_benchmark = false
device_allocator = "Futhark OpenCL runtime"
explicit_memory_preallocation = false
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
    [[ -x "$FUTTOP_BINARY" ]] || die "run setup.sh and build.sh before submitting benchmarks"
    [[ -d "$FUTTOP_SOURCE/.git" ]] || die "Futtop checkout is missing"
    [[ "$(git -C "$FUTTOP_SOURCE" rev-parse HEAD)" == "$SOURCE_REVISION" ]] ||
        die "Futtop checkout is not at the pinned revision"
    git -C "$FUTTOP_SOURCE" apply --reverse --check \
        "$FUTTOP_BENCH_DIR/patches/disable_vtr_output.patch" >/dev/null 2>&1 ||
        die "Futtop output patch is missing"

    SUBMISSION_ACCOUNT="${ACCOUNT:-${SLURM_JOB_ACCOUNT:-}}"
    ACCOUNT_ARGUMENT=()
    [[ -z "$SUBMISSION_ACCOUNT" ]] || ACCOUNT_ARGUMENT=(-A "$SUBMISSION_ACCOUNT")
    PARTITION="${PARTITION:-gpu}"
    CPUS="${CPUS:-72}"
    WALLTIME="${WALLTIME:-03:00:00}"
    ARRAY_SPEC="${TASK_INDEX:-0-10}"
    LOG_ROOT="$FUTTOP_BENCH_DIR/.slurm"
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
                --array="$ARRAY_SPEC" -J "futtop-traeff-${kind}" \
                -o "$LOG_DIR/traeff-${kind}-%A_%a.out" \
                --export=ALL,FUTTOP_RUN_ID="$RUN_ID",FUTTOP_MEASUREMENT="$kind",FUTTOP_BENCH_DIR="$FUTTOP_BENCH_DIR",TOPAX_ROOT="$TOPAX_ROOT" \
                "$SCRIPT_PATH"
        )"
        echo "Submitted $kind array: $job_id"
    done
    exit 0
fi

RUN_ID="${FUTTOP_RUN_ID:?}"
MEASUREMENT="${FUTTOP_MEASUREMENT:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"
EX="${MESH_X[$INDEX]}"
EY="${MESH_Y[$INDEX]}"
EZ="${MESH_Z[$INDEX]}"
CX=$((EX / 8))
CY=$((EY / 8))
CZ=$((EZ / 8))
MESH="${EX}x${EY}x${EZ}"
OUTPUT_DIR="$TOPAX_ROOT/benchmarks/results/traeff_2023/futtop/gpu/$MESH/$RUN_ID"
LOCK="$(acquire_result_lock "$OUTPUT_DIR" "$MEASUREMENT")"
SCRATCH="${SLURM_TMPDIR:-/tmp}/futtop-traeff-${SLURM_JOB_ID}-${INDEX}-${MEASUREMENT}"
mkdir -p "$SCRATCH"

cleanup() {
    stop_gpu_sampler
    rm -f "$OUTPUT_DIR/.${MEASUREMENT}.$$.csv" "$OUTPUT_DIR/.configuration.$$.toml"
    rm -rf "$LOCK" "$SCRATCH"
}
trap cleanup EXIT INT TERM

[[ -x "$FUTTOP_BINARY" ]] || die "Futtop executable is missing"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-72}"
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
    "$FUTTOP_BINARY"
    -x "$CX"
    -y "$CY"
    -z "$CZ"
    -v 0.1
    -r 1.5
    -i "$MAX_ITERATIONS"
)

cd "$SCRATCH"
set +e
/usr/bin/time -v -o "$TIME_FILE" "${COMMAND[@]}" > "$RAW" 2>&1
STATUS=$?
set -e
stop_gpu_sampler
if (( STATUS != 0 )); then
    cat "$RAW" >&2
    exit "$STATUS"
fi

parse_runtime "$RAW" "$ITERATION_FILE"
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

