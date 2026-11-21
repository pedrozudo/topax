#!/usr/bin/env bash
set -euo pipefail

# Measure steady-state host and device memory for the Topax Träff benchmark.
# Each array task runs the natural and initialization-cleanup variants
# sequentially so both measurements use the same node and GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
# shellcheck disable=SC1091
source "$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/scripts/common.sh"

MESH_X=(64 80 96 128 160 192 256 320 400 512 640)
MESH_Y=(32 40 48 64 80 96 128 160 200 256 320)
MESH_Z=(32 40 48 64 80 96 128 160 200 256 320)

usage() {
    cat <<'EOF'
Usage: traeff_gpu_steady_memory.sh [--run-id ID]

Measure phase-resolved host and GPU memory for the complete Träff L4-C2
sweep. Each mesh is measured both with and without explicitly releasing
initialization-only host data.
EOF
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    RUN_ID=""
    while (( $# )); do
        case "$1" in
            --run-id) RUN_ID="${2:?}"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) topax_die "unknown argument: $1" ;;
        esac
    done

    ACCOUNT="${ACCOUNT:-naiss2026-4-1571-gpu}"
    PARTITION="${PARTITION:-gpu}"
    CPUS="${CPUS:-72}"
    WALLTIME="${WALLTIME:-01:00:00}"
    LOG_ROOT="$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/scripts/.slurm"
    RUN_ID="$(topax_prepare_run_id "$RUN_ID" "$LOG_ROOT")"
    LOG_DIR="$LOG_ROOT/$RUN_ID"

    job_id="$(
        sbatch --parsable \
            -A "$ACCOUNT" -p "$PARTITION" -N 1 --ntasks=1 \
            --cpus-per-task="$CPUS" --gpus=1 -t "$WALLTIME" \
            --array=0-10 -J topax-traeff-steady-memory \
            -o "$LOG_DIR/traeff-steady-memory-%A_%a.out" \
            --export=ALL,TOPAX_RUN_ID="$RUN_ID",TOPAX_ROOT="$TOPAX_ROOT" \
            "$0"
    )"
    echo "Run ID: $RUN_ID"
    echo "Submitted steady-memory array: $job_id"
    exit 0
fi

RUN_ID="${TOPAX_RUN_ID:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"
EX="${MESH_X[$INDEX]}"
EY="${MESH_Y[$INDEX]}"
EZ="${MESH_Z[$INDEX]}"
MESH="${EX}x${EY}x${EZ}"
MAX_ITERATIONS=20
CPU_SAMPLE_INTERVAL="${CPU_SAMPLE_INTERVAL:-0.2}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-0.2}"
OUTPUT_DIR="$(
    topax_result_dir "$TOPAX_ROOT" traeff_2023 gpu "$MESH" "$RUN_ID"
)"
LOCK="$(topax_acquire_result_lock "$OUTPUT_DIR" steady_memory)"
SCRATCH="${SLURM_TMPDIR:-/tmp}/topax-traeff-steady-${SLURM_JOB_ID}-${INDEX}"
mkdir -p "$SCRATCH"

GPU_SAMPLER_PID=""
CPU_SAMPLER_PID=""
PHASE_WATCHER_PID=""
APP_WRAPPER_PID=""

stop_helpers() {
    local helper_pid
    for helper_pid in \
        "$PHASE_WATCHER_PID" "$CPU_SAMPLER_PID" "$GPU_SAMPLER_PID"; do
        if [[ -n "$helper_pid" ]]; then
            kill "$helper_pid" 2>/dev/null || true
            wait "$helper_pid" 2>/dev/null || true
        fi
    done
    GPU_SAMPLER_PID=""
    CPU_SAMPLER_PID=""
    PHASE_WATCHER_PID=""
}

cleanup() {
    local child_pid=""
    stop_helpers
    if [[ -n "$APP_WRAPPER_PID" ]]; then
        child_pid="$(pgrep -P "$APP_WRAPPER_PID" 2>/dev/null | head -n 1 || true)"
        [[ -z "$child_pid" ]] || kill "$child_pid" 2>/dev/null || true
        kill "$APP_WRAPPER_PID" 2>/dev/null || true
        [[ -z "$child_pid" ]] || wait "$child_pid" 2>/dev/null || true
        wait "$APP_WRAPPER_PID" 2>/dev/null || true
    fi
    rm -f "$OUTPUT_DIR/.steady_memory.$$.csv"
    rm -rf "$LOCK" "$SCRATCH"
}
trap cleanup EXIT INT TERM

GPU_ENV="${TOPAX_GPU_ENV:-$HOME/envs/topax-gpu.sh}"
[[ -f "$GPU_ENV" ]] || topax_die "GPU environment script not found"
# shellcheck disable=SC1090
source "$GPU_ENV"
export LC_ALL=C
unset TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_MEM_FRACTION \
    XLA_CLIENT_MEM_FRACTION || true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=address

PYTHON_BIN="${PYTHON:-python}"
command -v "$PYTHON_BIN" >/dev/null ||
    topax_die "Python executable not found: $PYTHON_BIN"

gpu_used_mib() {
    local device="${CUDA_VISIBLE_DEVICES:-0}"
    device="${device%%,*}"
    nvidia-smi -i "$device" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null |
        head -n 1 | tr -d '[:space:]'
}

write_configuration() {
    local temporary="$OUTPUT_DIR/.configuration.$$.toml"
    cat > "$temporary" <<EOF
schema_version = 2
benchmark = "traeff_2023"
implementation = "topax"
backend = "gpu"
elements = [$EX, $EY, $EZ]
resolution = "not_applicable"
levels = 4
galerkin_crossover_level = 2
smoothing_steps = 1
fine_grid_slab_size = 12
galerkin_slab_size = 4
precision = "float64"
physics_tolerance = 1e-5
filter_tolerance = "not_applicable"

[measurements.steady_memory]
design_iteration_limit = $MAX_ITERATIONS
full_benchmark = true
variants = ["natural", "release_initialization_memory"]
host_metric = "sampled Python process RSS"
host_sampling_interval_seconds = $CPU_SAMPLE_INTERVAL
gpu_sampling_interval_seconds = $GPU_SAMPLE_INTERVAL
gpu_baseline_corrected = true
steady_phase = "post-warmup design iterations"
jax_preallocate = false
jax_allocator = "address"
EOF
    topax_install_configuration "$OUTPUT_DIR" "$temporary"
}

summarize_variant() {
    local label="$1"
    local release="$2"
    local variant_dir="$3"
    local status="$4"
    local baseline_mib="$5"

    python3 - \
        "$label" "$release" "$status" "$baseline_mib" \
        "$variant_dir/runtime.csv" "$variant_dir/time.txt" \
        "$variant_dir/gpu.csv" "$variant_dir/cpu.csv" \
        "$variant_dir/phases.csv" "$variant_dir/summary.csv" <<'PY'
import csv
import sys
from pathlib import Path

(
    label,
    release,
    status,
    baseline_mib,
    runtime_path,
    time_path,
    gpu_path,
    cpu_path,
    phase_path,
    output_path,
) = sys.argv[1:]


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("memory measurement contains no phase samples")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def samples(path, value_field):
    result = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                result.append(
                    (float(row["timestamp_epoch_s"]), float(row[value_field]))
                )
            except (KeyError, TypeError, ValueError):
                pass
    return result


def in_phase(values, start, end):
    return [value for timestamp, value in values if start <= timestamp <= end]


phases = {}
with Path(phase_path).open(newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            phases[row["event"]] = float(row["timestamp_epoch_s"])
        except (KeyError, TypeError, ValueError):
            pass

required_phases = ("app_start", "design_start", "design_end", "app_end")
missing_phases = [phase for phase in required_phases if phase not in phases]
if missing_phases:
    raise RuntimeError(f"missing phase markers: {', '.join(missing_phases)}")

gpu = samples(gpu_path, "used_mib")
cpu = samples(cpu_path, "rss_kib")
setup_gpu = in_phase(gpu, phases["app_start"], phases["design_start"])
steady_gpu = in_phase(gpu, phases["design_start"], phases["design_end"])
setup_cpu = in_phase(cpu, phases["app_start"], phases["design_start"])
steady_cpu = in_phase(cpu, phases["design_start"], phases["design_end"])
if not all((setup_gpu, steady_gpu, setup_cpu, steady_cpu)):
    raise RuntimeError("one or more memory phases contain no samples")

with Path(runtime_path).open(newline="") as handle:
    iterations = sum(1 for _ in csv.DictReader(handle))

peak_rss_kib = None
for line in Path(time_path).read_text(errors="replace").splitlines():
    if "Maximum resident set size (kbytes):" in line:
        peak_rss_kib = int(line.rsplit(":", 1)[1].strip())
        break
if peak_rss_kib is None:
    raise RuntimeError("maximum resident set size is unavailable")

baseline = float(baseline_mib)
host_median = percentile(steady_cpu, 0.5) / 1024**2
host_p95 = percentile(steady_cpu, 0.95) / 1024**2
gpu_median = (percentile(steady_gpu, 0.5) - baseline) / 1024
gpu_p95 = (percentile(steady_gpu, 0.95) - baseline) / 1024

row = {
    "variant": label,
    "release_initialization_memory": int(release),
    "design_iterations_completed": iterations,
    "application_exit_status": int(status),
    "host_process_peak_rss_gib": peak_rss_kib / 1024**2,
    "host_setup_peak_rss_gib": max(setup_cpu) / 1024**2,
    "host_steady_samples": len(steady_cpu),
    "host_steady_rss_median_gib": host_median,
    "host_steady_rss_p95_gib": host_p95,
    "host_steady_rss_peak_gib": max(steady_cpu) / 1024**2,
    "gpu_baseline_gib": baseline / 1024,
    "gpu_setup_peak_delta_gib": (max(setup_gpu) - baseline) / 1024,
    "gpu_steady_samples": len(steady_gpu),
    "gpu_steady_delta_median_gib": gpu_median,
    "gpu_steady_delta_p95_gib": gpu_p95,
    "gpu_steady_delta_peak_gib": (max(steady_gpu) - baseline) / 1024,
    "combined_steady_median_gib": host_median + gpu_median,
    "combined_steady_p95_gib": host_p95 + gpu_p95,
}

with Path(output_path).open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=row)
    writer.writeheader()
    writer.writerow(row)
PY
}

run_variant() {
    local label="$1"
    local release="$2"
    local variant_dir="$SCRATCH/$label"
    local raw="$variant_dir/output.txt"
    local time_file="$variant_dir/time.txt"
    local gpu_file="$variant_dir/gpu.csv"
    local cpu_file="$variant_dir/cpu.csv"
    local phase_file="$variant_dir/phases.csv"
    local runtime_file="$variant_dir/runtime.csv"
    local baseline_mib
    local app_pid=""
    local status
    local -a release_args=()
    mkdir -p "$variant_dir"

    if [[ "$release" == 1 ]]; then
        release_args+=(--release-initialization-memory)
    fi

    baseline_mib="$(gpu_used_mib || true)"
    [[ "$baseline_mib" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        topax_die "GPU baseline is unavailable for $MESH/$label"

    echo "timestamp_epoch_s,used_mib" > "$gpu_file"
    echo "timestamp_epoch_s,rss_kib" > "$cpu_file"
    echo "event,timestamp_epoch_s" > "$phase_file"

    (
        while true; do
            local used
            used="$(gpu_used_mib || true)"
            if [[ "$used" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                printf '%s,%s\n' "$EPOCHREALTIME" "$used" >> "$gpu_file"
            fi
            sleep "$GPU_SAMPLE_INTERVAL"
        done
    ) &
    GPU_SAMPLER_PID=$!

    set +e
    /usr/bin/time -v -o "$time_file" \
        "$PYTHON_BIN" -u \
        "$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py" \
        --elements "$EX" "$EY" "$EZ" \
        --levels 4 \
        --galerkin-crossover-level 2 \
        --smoothing-steps 1 \
        --max-iterations "$MAX_ITERATIONS" \
        --runtime-csv "$runtime_file" \
        "${release_args[@]}" \
        > "$raw" 2>&1 &
    APP_WRAPPER_PID=$!
    set -e

    for _ in $(seq 1 500); do
        app_pid="$(pgrep -P "$APP_WRAPPER_PID" 2>/dev/null | head -n 1 || true)"
        if [[ -n "$app_pid" ]]; then
            break
        fi
        if ! kill -0 "$APP_WRAPPER_PID" 2>/dev/null; then
            break
        fi
        sleep 0.02
    done
    [[ -n "$app_pid" ]] || topax_die "could not resolve the Python PID"
    printf 'app_start,%s\n' "$EPOCHREALTIME" >> "$phase_file"

    (
        while kill -0 "$app_pid" 2>/dev/null; do
            local rss_kib
            rss_kib="$(
                awk '$1 == "VmRSS:" {print $2; exit}' \
                    "/proc/$app_pid/status" 2>/dev/null || true
            )"
            if [[ "$rss_kib" =~ ^[0-9]+$ ]]; then
                printf '%s,%s\n' "$EPOCHREALTIME" "$rss_kib" >> "$cpu_file"
            fi
            sleep "$CPU_SAMPLE_INTERVAL"
        done
    ) &
    CPU_SAMPLER_PID=$!

    (
        local design_started=0
        while kill -0 "$app_pid" 2>/dev/null; do
            if [[ "$design_started" == 0 ]] \
                && grep -q '^dummy execution:' "$raw" 2>/dev/null; then
                printf 'design_start,%s\n' "$EPOCHREALTIME" >> "$phase_file"
                design_started=1
            fi
            if [[ "$design_started" == 1 ]] \
                && grep -q '^FINAL RESULT$' "$raw" 2>/dev/null; then
                printf 'design_end,%s\n' "$EPOCHREALTIME" >> "$phase_file"
                break
            fi
            sleep 0.05
        done
    ) &
    PHASE_WATCHER_PID=$!

    set +e
    wait "$APP_WRAPPER_PID"
    status=$?
    set -e
    APP_WRAPPER_PID=""
    printf 'app_end,%s\n' "$EPOCHREALTIME" >> "$phase_file"
    if ! grep -q '^design_end,' "$phase_file" \
        && grep -q '^FINAL RESULT$' "$raw" 2>/dev/null; then
        printf 'design_end,%s\n' "$EPOCHREALTIME" >> "$phase_file"
    fi
    stop_helpers

    if (( status != 0 )); then
        cat "$raw" >&2
        topax_die "$MESH/$label exited with status $status"
    fi

    summarize_variant \
        "$label" "$release" "$variant_dir" "$status" "$baseline_mib"
}

echo "Measuring Topax Träff steady memory for $MESH"
run_variant natural 0
run_variant release_initialization_memory 1

TEMP_RESULT="$OUTPUT_DIR/.steady_memory.$$.csv"
head -n 1 "$SCRATCH/natural/summary.csv" > "$TEMP_RESULT"
tail -n +2 "$SCRATCH/natural/summary.csv" >> "$TEMP_RESULT"
tail -n +2 "$SCRATCH/release_initialization_memory/summary.csv" \
    >> "$TEMP_RESULT"

write_configuration
mv "$TEMP_RESULT" "$OUTPUT_DIR/steady_memory.csv"
echo "Wrote $OUTPUT_DIR/steady_memory.csv"
