#!/usr/bin/env bash

# Shared execution and measurement logic for local Topax GPU benchmarks.

TOPAX_LOCAL_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$TOPAX_LOCAL_COMMON_DIR/common.sh"

topax_local_usage() {
    cat <<EOF
Usage: $(basename "$TOPAX_LOCAL_LAUNCHER") [options]

Run the six $TOPAX_LOCAL_TITLE meshes sequentially on the local CUDA GPU.

Options:
  --run-id ID          UTC ID in YYYYMMDDTHHMMSSZ format (default: now)
  --measurement KIND   runtime, memory, or both (default: both)
  --case MESH          run one mesh instead of all six
  -h, --help            show this message

The Python executable is selected with the PYTHON environment variable.
EOF
}

topax_local_case_index() {
    local requested="$1"
    local index
    for index in "${!TOPAX_LOCAL_CASES[@]}"; do
        if [[ "${TOPAX_LOCAL_CASES[$index]}" == "$requested" ]]; then
            printf '%s\n' "$index"
            return
        fi
    done
    topax_die "unknown case: $requested"
}

topax_local_gpu_name() {
    local device="${CUDA_VISIBLE_DEVICES:-0}"
    device="${device%%,*}"
    nvidia-smi -i "$device" --query-gpu=name --format=csv,noheader |
        head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

topax_local_install_configuration() {
    local output_dir="$1"
    local ex="$2"
    local ey="$3"
    local levels="$4"
    local gpu_name="$5"
    local temporary="$output_dir/.configuration.$$.toml"

    topax_local_write_problem_configuration \
        "$temporary" "$ex" "$ey" "$levels" "$gpu_name"
    topax_append_measurement_configuration \
        "$temporary" gpu 10000 10 true false true
    topax_install_configuration "$output_dir" "$temporary"
}

topax_local_run_measurement() (
    set -euo pipefail
    export LC_ALL=C

    local index="$1"
    local measurement="$2"
    local run_id="$3"
    local log_dir="$4"
    local python_bin="$5"
    local gpu_name="$6"
    local case_name="${TOPAX_LOCAL_CASES[$index]}"
    local ex="${TOPAX_LOCAL_MESH_X[$index]}"
    local ey="${TOPAX_LOCAL_MESH_Y[$index]}"
    local levels="${TOPAX_LOCAL_LEVELS[$index]}"
    local output_dir
    output_dir="$(
        topax_result_dir \
            "$TOPAX_ROOT" "$TOPAX_LOCAL_BENCHMARK" gpu "$case_name" "$run_id"
    )"
    local lock
    lock="$(topax_acquire_result_lock "$output_dir" "$measurement")"
    local scratch
    scratch="$(mktemp -d "/tmp/topax-${TOPAX_LOCAL_BENCHMARK}-${case_name}-${measurement}.XXXXXX")"
    local raw="$log_dir/${case_name}-${measurement}.log"
    local time_file="$scratch/time.txt"
    local gpu_file="$scratch/gpu.csv"
    local iteration_file="$scratch/runtime.csv"
    local artifact_dir="$scratch/artifacts"
    local temporary_result="$output_dir/.${measurement}.$$.csv"

    # shellcheck disable=SC2317
    cleanup() {
        topax_stop_gpu_sampler
        rm -f "$temporary_result" "$output_dir/.configuration.$$.toml"
        rm -rf "$lock" "$scratch"
    }
    trap cleanup EXIT INT TERM
    mkdir -p "$artifact_dir"

    local max_iterations
    if [[ "$measurement" == memory ]]; then
        unset TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_MEM_FRACTION \
            XLA_CLIENT_MEM_FRACTION || true
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export XLA_PYTHON_CLIENT_ALLOCATOR=address
        max_iterations=10
        topax_start_gpu_sampler "$gpu_file" "${GPU_SAMPLE_INTERVAL:-0.2}"
    else
        unset XLA_PYTHON_CLIENT_ALLOCATOR XLA_CLIENT_MEM_FRACTION || true
        export XLA_PYTHON_CLIENT_PREALLOCATE=true
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
        max_iterations=10000
        [[ ! -e "$output_dir/artifacts" ]] ||
            topax_die "artifacts already exist: $output_dir/artifacts"
    fi

    topax_local_build_command \
        "$python_bin" "$case_name" "$levels" "$max_iterations" \
        "$iteration_file" "$artifact_dir"

    echo
    echo "$TOPAX_LOCAL_TITLE | $case_name | $measurement"
    printf 'Command:'
    printf ' %q' "${TOPAX_LOCAL_COMMAND[@]}"
    echo

    set +e
    /usr/bin/time -v -o "$time_file" "${TOPAX_LOCAL_COMMAND[@]}" \
        2>&1 | tee "$raw"
    local status="${PIPESTATUS[0]}"
    set -e
    topax_stop_gpu_sampler
    if (( status != 0 )); then
        exit "$status"
    fi

    if [[ "$measurement" == memory ]]; then
        topax_write_memory_csv \
            "$temporary_result" "$iteration_file" "$time_file" \
            "$gpu_file" "${TOPAX_GPU_BASELINE_MIB:-}"
    else
        mv "$iteration_file" "$temporary_result"
    fi

    topax_local_install_configuration \
        "$output_dir" "$ex" "$ey" "$levels" "$gpu_name"
    if [[ "$measurement" == runtime ]]; then
        mv "$artifact_dir" "$output_dir/artifacts"
    fi
    mv "$temporary_result" "$output_dir/$measurement.csv"
    echo "Wrote $output_dir/$measurement.csv"
)

topax_run_local_gpu_campaign() {
    local run_id=""
    local measurement="both"
    local selected_case=""
    while (( $# )); do
        case "$1" in
            --run-id) run_id="${2:?}"; shift 2 ;;
            --measurement) measurement="${2:?}"; shift 2 ;;
            --case) selected_case="${2:?}"; shift 2 ;;
            -h|--help) topax_local_usage; return ;;
            *) topax_die "unknown argument: $1" ;;
        esac
    done
    topax_validate_measurement "$measurement"

    local count="${#TOPAX_LOCAL_CASES[@]}"
    (( count == 6 )) || topax_die "the local campaign must define six cases"
    (( ${#TOPAX_LOCAL_MESH_X[@]} == count )) || topax_die "mesh-x array mismatch"
    (( ${#TOPAX_LOCAL_MESH_Y[@]} == count )) || topax_die "mesh-y array mismatch"
    (( ${#TOPAX_LOCAL_LEVELS[@]} == count )) || topax_die "level array mismatch"

    command -v nvidia-smi >/dev/null || topax_die "nvidia-smi not found"
    [[ -x /usr/bin/time ]] || topax_die "/usr/bin/time not found"
    local python_bin="${PYTHON:-python}"
    command -v "$python_bin" >/dev/null ||
        topax_die "Python executable not found: $python_bin"
    local gpu_name
    gpu_name="$(topax_local_gpu_name)"
    [[ -n "$gpu_name" ]] || topax_die "could not identify the CUDA GPU"

    local run_root="${TOPAX_LOCAL_RUN_ROOT:-$TOPAX_LOCAL_SCRIPT_DIR/.runs/$TOPAX_LOCAL_BENCHMARK}"
    run_id="$(topax_prepare_run_id "$run_id" "$run_root")"
    local log_dir="$run_root/$run_id"
    local indices=()
    if [[ -n "$selected_case" ]]; then
        indices+=("$(topax_local_case_index "$selected_case")")
    else
        indices=("${!TOPAX_LOCAL_CASES[@]}")
    fi
    local measurements=("$measurement")
    [[ "$measurement" != both ]] || measurements=(runtime memory)

    echo "Benchmark: $TOPAX_LOCAL_TITLE"
    echo "Run ID:    $run_id"
    echo "GPU:       $gpu_name"
    echo "Python:    $(command -v "$python_bin")"

    local index kind
    for index in "${indices[@]}"; do
        for kind in "${measurements[@]}"; do
            topax_local_run_measurement \
                "$index" "$kind" "$run_id" "$log_dir" \
                "$python_bin" "$gpu_name"
        done
    done
}
