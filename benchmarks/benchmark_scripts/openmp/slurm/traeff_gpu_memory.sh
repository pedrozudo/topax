#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
BENCH_DIR="${OPENMP_BENCH_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$BENCH_DIR/../../.." && pwd)}"
OPENMP_SOURCE="$BENCH_DIR/TopOpt-in-OpenMP-GPU"
OPENMP_BINARY="$BENCH_DIR/build/top3d"
SOURCE_REVISION="b46ed453ad5a05818b563e540ce176c9d85c6fe2"

MESH_X=(64 80 96 128 160 192 256 320 400 512 640)
MESH_Y=(32 40 48 64 80 96 128 160 200 256 320)
MESH_Z=(32 40 48 64 80 96 128 160 200 256 320)

die() {
    echo "ERROR: $*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
Usage: traeff_gpu_memory.sh [options]

Options:
  --run-id ID       UTC ID in YYYYMMDDTHHMMSSZ format (default: now)
  --task-index N    submit only mesh index N, from 0 through 10
EOF
}

library_dir() {
    if [[ -d "$1/lib64" ]]; then
        printf '%s\n' "$1/lib64"
    elif [[ -d "$1/lib" ]]; then
        printf '%s\n' "$1/lib"
    else
        die "library directory not found below $1"
    fi
}

acquire_result_lock() {
    local output_dir="$1"
    local result="$output_dir/memory.csv"
    local lock="$output_dir/.memory.lock"

    mkdir -p "$output_dir"
    [[ ! -e "$result" ]] || die "result already exists: $result"
    mkdir "$lock" 2>/dev/null || die "another memory measurement is using $output_dir"
    printf '%s\n' "$lock"
}

gpu_used_mib() {
    local device="${CUDA_VISIBLE_DEVICES:-0}"
    device="${device%%,*}"
    nvidia-smi -i "$device" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null |
        head -n 1 | tr -d '[:space:]'
}

stop_helpers() {
    local pid
    for pid in "${WATCHDOG_PID:-}" "${CPU_SAMPLER_PID:-}" "${GPU_SAMPLER_PID:-}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    unset WATCHDOG_PID CPU_SAMPLER_PID GPU_SAMPLER_PID
}

write_memory_csv() {
    local output="$1"
    local raw="$2"
    local time_file="$3"
    local gpu_file="$4"
    local cpu_file="$5"
    local baseline_mib="$6"
    local app_start="$7"
    local terminated="$8"
    local app_status="$9"

    python3 - "$output" "$raw" "$time_file" "$gpu_file" "$cpu_file" \
        "$baseline_mib" "$app_start" "$terminated" "$app_status" \
        "$STEADY_STATE_DELAY" "$MEASUREMENT_DURATION" <<'PY'
import csv
import re
import statistics
import sys
from pathlib import Path

(
    output_path,
    raw_path,
    time_path,
    gpu_path,
    cpu_path,
    baseline_text,
    app_start_text,
    terminated_text,
    status_text,
    delay_text,
    duration_text,
) = sys.argv[1:]

output_path = Path(output_path)
raw = Path(raw_path).read_text(errors="replace")
time_text = Path(time_path).read_text(errors="replace")
baseline_mib = float(baseline_text)
app_start = float(app_start_text)
terminated = int(terminated_text)
application_status = int(status_text)
delay = float(delay_text)
duration = float(duration_text)


def read_samples(path, value_field):
    samples = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                samples.append(
                    (float(row["timestamp_epoch_s"]), float(row[value_field]))
                )
            except (KeyError, ValueError):
                pass
    return samples


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise SystemExit("steady-state memory sample set is empty")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


gpu_samples = read_samples(gpu_path, "used_mib")
cpu_samples = read_samples(cpu_path, "rss_kib")
threshold = app_start + delay
gpu_values = [value for _, value in gpu_samples]
cpu_values = [value for _, value in cpu_samples]
gpu_steady = [value for timestamp, value in gpu_samples if timestamp >= threshold]
cpu_steady = [value for timestamp, value in cpu_samples if timestamp >= threshold]
if not gpu_values or not cpu_values:
    raise SystemExit("memory samples are unavailable")
if not gpu_steady or not cpu_steady:
    raise SystemExit("no memory samples remain after the steady-state delay")

rss_match = re.search(
    r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text
)
if not rss_match:
    raise SystemExit("GNU time did not report maximum resident memory")
host_peak_kib = int(rss_match.group(1))
iterations = len(re.findall(r"^It\.:\s*\d+", raw, re.MULTILINE))

gpu_peak_mib = max(gpu_values)
gpu_median_delta_mib = statistics.median(gpu_steady) - baseline_mib
gpu_p95_delta_mib = percentile(gpu_steady, 0.95) - baseline_mib
host_median_gib = statistics.median(cpu_steady) / 1024**2
host_p95_gib = percentile(cpu_steady, 0.95) / 1024**2
gpu_median_gib = gpu_median_delta_mib / 1024
gpu_p95_gib = gpu_p95_delta_mib / 1024

fields = [
    "design_iterations_completed",
    "application_exit_status",
    "measurement_window_terminated",
    "measurement_duration_s",
    "steady_state_delay_s",
    "host_peak_rss_gib",
    "host_sampled_peak_rss_gib",
    "host_steady_rss_median_gib",
    "host_steady_rss_p95_gib",
    "host_steady_samples",
    "gpu_baseline_gib",
    "gpu_peak_gib",
    "gpu_peak_above_baseline_gib",
    "gpu_steady_delta_median_gib",
    "gpu_steady_delta_p95_gib",
    "gpu_steady_samples",
    "combined_steady_median_gib",
    "combined_steady_p95_gib",
]
row = {
    "design_iterations_completed": iterations,
    "application_exit_status": application_status,
    "measurement_window_terminated": terminated,
    "measurement_duration_s": duration,
    "steady_state_delay_s": delay,
    "host_peak_rss_gib": host_peak_kib / 1024**2,
    "host_sampled_peak_rss_gib": max(cpu_values) / 1024**2,
    "host_steady_rss_median_gib": host_median_gib,
    "host_steady_rss_p95_gib": host_p95_gib,
    "host_steady_samples": len(cpu_steady),
    "gpu_baseline_gib": baseline_mib / 1024,
    "gpu_peak_gib": gpu_peak_mib / 1024,
    "gpu_peak_above_baseline_gib": (gpu_peak_mib - baseline_mib) / 1024,
    "gpu_steady_delta_median_gib": gpu_median_gib,
    "gpu_steady_delta_p95_gib": gpu_p95_gib,
    "gpu_steady_samples": len(gpu_steady),
    "combined_steady_median_gib": host_median_gib + gpu_median_gib,
    "combined_steady_p95_gib": host_p95_gib + gpu_p95_gib,
}
with output_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerow(row)
PY
}

write_configuration() {
    local output_dir="$1"
    local temporary="$output_dir/.configuration.$$.toml"
    local compiler_version

    compiler_version="$(awk 'NF {print; exit}' "$BENCH_DIR/build/compiler-version.txt")"
    compiler_version="${compiler_version//\"/\\\"}"
    cat > "$temporary" <<EOF
schema_version = 2
benchmark = "traeff_2023"
implementation = "openmp"
implementation_version = "0.9.1"
source_revision = "$SOURCE_REVISION"
backend = "openmp_target_cuda"
elements = [$EX, $EY, $EZ]
levels = 4
coarsenings = 3
coarse_operator = "multiresolution approximation"
cpu_resident_coarse_levels = 2
coarse_solver = "CHOLMOD"
precision = "mixed_float32_float64"
physics_tolerance = 1e-5
physics_max_iterations = 999
filter = "linear density filter"
filter_radius_elements = 1.5
volume_fraction = 0.1
simp_penalty = 3
compiler = "$compiler_version"
compiler_module = "$COMPILER_MODULE"
gpu_architecture = "cc90"
offload_memory_mode = "separate"
cpu_threads = ${SLURM_CPUS_PER_TASK:-12}

[measurements.memory]
design_iteration_limit = 20
full_benchmark = false
bounded_measurement = true
measurement_duration_seconds = $MEASUREMENT_DURATION
steady_state_delay_seconds = $STEADY_STATE_DELAY
gpu_sampling_interval_seconds = $GPU_SAMPLE_INTERVAL
host_sampling_interval_seconds = $CPU_SAMPLE_INTERVAL
host_metric = "application process RSS"
EOF

    if [[ -e "$output_dir/configuration.toml" ]]; then
        cmp -s "$temporary" "$output_dir/configuration.toml" || {
            rm -f "$temporary"
            die "configuration mismatch in $output_dir"
        }
        rm -f "$temporary"
    else
        mv "$temporary" "$output_dir/configuration.toml"
    fi
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    RUN_ID=""
    TASK_INDEX=""
    while (( $# )); do
        case "$1" in
            --run-id) RUN_ID="${2:?}"; shift 2 ;;
            --task-index) TASK_INDEX="${2:?}"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    if [[ -n "$TASK_INDEX" ]]; then
        [[ "$TASK_INDEX" =~ ^([0-9]|10)$ ]] ||
            die "--task-index must be an integer from 0 through 10"
    fi
    [[ -n "${ACCOUNT:-}" ]] || die "set ACCOUNT"
    [[ -n "${COMPILER_MODULE:-}" ]] || die "set COMPILER_MODULE"
    [[ -n "${SUITESPARSE_ROOT:-}" ]] || die "set SUITESPARSE_ROOT"
    [[ -n "${OPENBLAS_ROOT:-}" ]] || die "set OPENBLAS_ROOT"
    [[ -n "${GCC_RUNTIME_ROOT:-}" ]] || die "set GCC_RUNTIME_ROOT"
    [[ -x "$OPENMP_BINARY" ]] || die "run setup.sh and build.sh before submitting"
    [[ -f "$BENCH_DIR/build/build.toml" ]] || die "build metadata is missing"
    [[ -d "$OPENMP_SOURCE/.git" ]] || die "OpenMP checkout is missing"
    [[ "$(git -C "$OPENMP_SOURCE" rev-parse HEAD)" == "$SOURCE_REVISION" ]] ||
        die "OpenMP checkout is not at the pinned revision"
    git -C "$OPENMP_SOURCE" diff --quiet ||
        die "OpenMP checkout contains tracked source modifications"

    if [[ -z "$RUN_ID" ]]; then
        RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    [[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] ||
        die "run ID must use the UTC format YYYYMMDDTHHMMSSZ"
    PARTITION="${PARTITION:-gpu}"
    CPUS="${CPUS:-12}"
    MEMORY="${MEMORY:-64G}"
    WALLTIME="${WALLTIME:-00:10:00}"
    ARRAY_SPEC="${TASK_INDEX:-0-10}"
    GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-0.2}"
    CPU_SAMPLE_INTERVAL="${CPU_SAMPLE_INTERVAL:-0.2}"
    MEASUREMENT_DURATION="${MEASUREMENT_DURATION:-180}"
    STEADY_STATE_DELAY="${STEADY_STATE_DELAY:-30}"
    LOG_DIR="$BENCH_DIR/.slurm/$RUN_ID"
    mkdir -p "$LOG_DIR"

    JOB_ID="$(
        sbatch --parsable \
            -A "$ACCOUNT" -p "$PARTITION" -N 1 --ntasks=1 \
            --cpus-per-task="$CPUS" --mem="$MEMORY" --gpus=1 \
            -t "$WALLTIME" --array="$ARRAY_SPEC" \
            -J "openmp-traeff-memory" \
            -o "$LOG_DIR/traeff-memory-%A_%a.out" \
            --export=ALL,OPENMP_RUN_ID="$RUN_ID",OPENMP_BENCH_DIR="$BENCH_DIR",TOPAX_ROOT="$TOPAX_ROOT",COMPILER_MODULE="$COMPILER_MODULE",SUITESPARSE_ROOT="$SUITESPARSE_ROOT",OPENBLAS_ROOT="$OPENBLAS_ROOT",GCC_RUNTIME_ROOT="$GCC_RUNTIME_ROOT",GPU_SAMPLE_INTERVAL="$GPU_SAMPLE_INTERVAL",CPU_SAMPLE_INTERVAL="$CPU_SAMPLE_INTERVAL",MEASUREMENT_DURATION="$MEASUREMENT_DURATION",STEADY_STATE_DELAY="$STEADY_STATE_DELAY" \
            "$SCRIPT_PATH"
    )"
    echo "Run ID: $RUN_ID"
    echo "Submitted memory array: $JOB_ID"
    exit 0
fi

RUN_ID="${OPENMP_RUN_ID:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"
EX="${MESH_X[$INDEX]}"
EY="${MESH_Y[$INDEX]}"
EZ="${MESH_Z[$INDEX]}"
CX=$((EX / 8))
CY=$((EY / 8))
CZ=$((EZ / 8))
MESH="${EX}x${EY}x${EZ}"
OUTPUT_DIR="$TOPAX_ROOT/benchmarks/results/traeff_2023/openmp/gpu/$MESH/$RUN_ID"
LOCK="$(acquire_result_lock "$OUTPUT_DIR")"
SCRATCH="${SLURM_TMPDIR:-/tmp}/openmp-traeff-${SLURM_JOB_ID}-${INDEX}"
mkdir -p "$SCRATCH"

cleanup() {
    stop_helpers
    rm -f "$OUTPUT_DIR/.memory.$$.csv" "$OUTPUT_DIR/.configuration.$$.toml"
    rm -rf "$LOCK" "$SCRATCH"
}
trap cleanup EXIT INT TERM

[[ "$(uname -m)" == aarch64 ]] || die "memory worker must run on AArch64"
module --force purge
module load "$COMPILER_MODULE"
SUITESPARSE_LIBDIR="$(library_dir "$SUITESPARSE_ROOT")"
OPENBLAS_LIBDIR="$(library_dir "$OPENBLAS_ROOT")"
export LD_LIBRARY_PATH="$SUITESPARSE_LIBDIR:$OPENBLAS_LIBDIR:$GCC_RUNTIME_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_TARGET_OFFLOAD=MANDATORY
export LC_ALL=C

RAW="$SCRATCH/output.txt"
TIME_FILE="$SCRATCH/time.txt"
GPU_FILE="$SCRATCH/gpu.csv"
CPU_FILE="$SCRATCH/cpu.csv"
STOP_FILE="$SCRATCH/measurement-ended"
echo "timestamp_epoch_s,used_mib" > "$GPU_FILE"
echo "timestamp_epoch_s,rss_kib" > "$CPU_FILE"

GPU_BASELINE_MIB="$(gpu_used_mib || true)"
[[ "$GPU_BASELINE_MIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "GPU baseline is unavailable"
(
    while true; do
        used="$(gpu_used_mib || true)"
        if [[ "$used" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            printf '%s,%s\n' "$EPOCHREALTIME" "$used" >> "$GPU_FILE"
        fi
        sleep "$GPU_SAMPLE_INTERVAL"
    done
) &
GPU_SAMPLER_PID=$!

cd "$SCRATCH"
set +e
/usr/bin/time -v -o "$TIME_FILE" \
    stdbuf -oL -eL "$OPENMP_BINARY" \
        -x "$CX" -y "$CY" -z "$CZ" -l 4 \
        -f 0.10 -r 1.5 -i 20 -v 1 -w 0 \
        > "$RAW" 2>&1 &
TIME_PID=$!
set -e

APP_PID=""
for _ in $(seq 1 500); do
    APP_PID="$(pgrep -P "$TIME_PID" -x top3d 2>/dev/null | head -n 1 || true)"
    [[ -z "$APP_PID" ]] || break
    kill -0 "$TIME_PID" 2>/dev/null || break
    sleep 0.02
done
[[ -n "$APP_PID" ]] || die "could not resolve the top3d process"
APP_START_EPOCH="$EPOCHREALTIME"

(
    while kill -0 "$APP_PID" 2>/dev/null; do
        rss="$(awk '$1 == "VmRSS:" {print $2; exit}' "/proc/$APP_PID/status" 2>/dev/null || true)"
        if [[ "$rss" =~ ^[0-9]+$ ]]; then
            printf '%s,%s\n' "$EPOCHREALTIME" "$rss" >> "$CPU_FILE"
        fi
        sleep "$CPU_SAMPLE_INTERVAL"
    done
) &
CPU_SAMPLER_PID=$!

(
    sleep "$MEASUREMENT_DURATION"
    if kill -0 "$APP_PID" 2>/dev/null; then
        : > "$STOP_FILE"
        kill -INT "$APP_PID" 2>/dev/null || true
        for _ in $(seq 1 150); do
            kill -0 "$APP_PID" 2>/dev/null || exit 0
            sleep 0.1
        done
        kill -TERM "$APP_PID" 2>/dev/null || true
        sleep 5
        kill -KILL "$APP_PID" 2>/dev/null || true
    fi
) &
WATCHDOG_PID=$!

set +e
wait "$TIME_PID"
APPLICATION_STATUS=$?
set -e
MEASUREMENT_WINDOW_TERMINATED=0
[[ ! -f "$STOP_FILE" ]] || MEASUREMENT_WINDOW_TERMINATED=1
stop_helpers

TEMP_RESULT="$OUTPUT_DIR/.memory.$$.csv"
write_memory_csv \
    "$TEMP_RESULT" "$RAW" "$TIME_FILE" "$GPU_FILE" "$CPU_FILE" \
    "$GPU_BASELINE_MIB" "$APP_START_EPOCH" \
    "$MEASUREMENT_WINDOW_TERMINATED" "$APPLICATION_STATUS"
write_configuration "$OUTPUT_DIR"
mv "$TEMP_RESULT" "$OUTPUT_DIR/memory.csv"
echo "Wrote $OUTPUT_DIR/memory.csv"

if (( MEASUREMENT_WINDOW_TERMINATED == 0 && APPLICATION_STATUS != 0 )); then
    exit "$APPLICATION_STATUS"
fi

