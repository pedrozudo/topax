#!/usr/bin/env bash

topax_die() {
    echo "ERROR: $*" >&2
    exit 2
}

topax_validate_run_id() {
    local run_id="$1"
    [[ "$run_id" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] ||
        topax_die "run ID must use the UTC format YYYYMMDDTHHMMSSZ"
}

topax_prepare_run_id() {
    local requested="$1"
    local log_root="$2"
    local candidate

    mkdir -p "$log_root"
    if [[ -n "$requested" ]]; then
        topax_validate_run_id "$requested"
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

topax_validate_hierarchy() {
    local levels="$1"
    local crossover="$2"
    [[ "$levels" =~ ^[0-9]+$ ]] || topax_die "levels must be an integer"
    [[ "$crossover" =~ ^[0-9]+$ ]] ||
        topax_die "crossover level must be an integer"
    (( levels >= 2 )) || topax_die "levels must be at least 2"
    (( crossover >= 1 && crossover < levels )) ||
        topax_die "crossover level must satisfy 1 <= crossover < levels"
}

topax_validate_measurement() {
    case "$1" in
        runtime|memory|both) ;;
        *) topax_die "measurement must be runtime, memory, or both" ;;
    esac
}

topax_result_dir() {
    local root="$1"
    local benchmark="$2"
    local backend="$3"
    local mesh="$4"
    local run_id="$5"
    printf '%s/benchmarks/results/%s/topax/%s/%s/%s\n' \
        "$root" "$benchmark" "$backend" "$mesh" "$run_id"
}

topax_acquire_result_lock() {
    local output_dir="$1"
    local measurement="$2"
    local result="$output_dir/$measurement.csv"
    local lock="$output_dir/.$measurement.lock"

    mkdir -p "$output_dir"
    [[ ! -e "$result" ]] || topax_die "result already exists: $result"
    mkdir "$lock" 2>/dev/null ||
        topax_die "another $measurement measurement is using $output_dir"
    printf '%s\n' "$lock"
}

topax_install_configuration() {
    local output_dir="$1"
    local temporary="$2"
    local destination="$output_dir/configuration.toml"

    if [[ -e "$destination" ]]; then
        if ! cmp -s "$temporary" "$destination"; then
            rm -f "$temporary"
            topax_die "configuration mismatch in $output_dir"
        fi
        rm -f "$temporary"
    else
        mv "$temporary" "$destination"
    fi
}

topax_write_configuration() {
    local output_dir="$1"
    local benchmark="$2"
    local backend="$3"
    local mesh_x="$4"
    local mesh_y="$5"
    local mesh_z="$6"
    local resolution="$7"
    local levels="$8"
    local crossover="$9"
    local smoothing="${10}"
    local runtime_iterations="${11}"
    local memory_iterations="${12}"
    local physics_tolerance="${13}"
    local filter_tolerance="${14}"
    local runtime_full_benchmark="${15}"
    local memory_full_benchmark="${16}"
    local runtime_jax_preallocate="${17:-false}"
    local temporary="$output_dir/.configuration.$$.toml"

    cat > "$temporary" <<EOF
schema_version = 2
benchmark = "$benchmark"
implementation = "topax"
backend = "$backend"
elements = [$mesh_x, $mesh_y, $mesh_z]
resolution = "$resolution"
levels = $levels
galerkin_crossover_level = $crossover
smoothing_steps = $smoothing
fine_grid_slab_size = 12
galerkin_slab_size = 4
precision = "float64"
physics_tolerance = $physics_tolerance
filter_tolerance = $filter_tolerance
EOF

    if [[ "$benchmark" == wang_2025 ]]; then
        cat >> "$temporary" <<'EOF'
filter = "TOP3D_XL PDE sensitivity filter"
filter_discretization = "released TOP3D_XL element operator"
filter_radius_elements = 1.7320508075688772
equivalent_topax_helmholtz_radius_elements = 2.449489742783178
EOF
    fi

    topax_append_measurement_configuration \
        "$temporary" "$backend" "$runtime_iterations" "$memory_iterations" \
        "$runtime_full_benchmark" "$memory_full_benchmark" \
        "$runtime_jax_preallocate"

    topax_install_configuration "$output_dir" "$temporary"
}

topax_append_measurement_configuration() {
    local configuration="$1"
    local backend="$2"
    local runtime_iterations="$3"
    local memory_iterations="$4"
    local runtime_full_benchmark="$5"
    local memory_full_benchmark="$6"
    local runtime_jax_preallocate="${7:-false}"

    cat >> "$configuration" <<EOF

[measurements.runtime]
design_iteration_limit = $runtime_iterations
full_benchmark = $runtime_full_benchmark
EOF
    if [[ "$backend" == gpu ]]; then
        if [[ "$runtime_jax_preallocate" == true ]]; then
            cat >> "$configuration" <<'EOF'
jax_preallocate = true
jax_allocator = "default"
jax_memory_fraction = 0.75
EOF
        else
            cat >> "$configuration" <<'EOF'
jax_preallocate = false
jax_allocator = "address"
EOF
        fi
    fi

    cat >> "$configuration" <<EOF

[measurements.memory]
design_iteration_limit = $memory_iterations
full_benchmark = $memory_full_benchmark
EOF
    if [[ "$backend" == gpu ]]; then
        cat >> "$configuration" <<'EOF'
jax_preallocate = false
jax_allocator = "address"
EOF
    fi
}

topax_start_gpu_sampler() {
    local output="$1"
    local interval="$2"
    local device="${CUDA_VISIBLE_DEVICES:-0}"
    device="${device%%,*}"

    TOPAX_GPU_BASELINE_MIB="$(
        nvidia-smi -i "$device" --query-gpu=memory.used \
            --format=csv,noheader,nounits 2>/dev/null |
            head -n 1 | tr -d '[:space:]'
    )"
    [[ "$TOPAX_GPU_BASELINE_MIB" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        TOPAX_GPU_BASELINE_MIB=""

    echo "timestamp_epoch_s,used_mib" > "$output"
    (
        while true; do
            local used
            used="$(
                nvidia-smi -i "$device" --query-gpu=memory.used \
                    --format=csv,noheader,nounits 2>/dev/null |
                    head -n 1 | tr -d '[:space:]'
            )"
            if [[ "$used" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                printf '%s,%s\n' "$(date +%s.%N)" "$used" >> "$output"
            fi
            sleep "$interval"
        done
    ) &
    TOPAX_GPU_SAMPLER_PID=$!
    export TOPAX_GPU_BASELINE_MIB TOPAX_GPU_SAMPLER_PID
}

topax_stop_gpu_sampler() {
    if [[ -n "${TOPAX_GPU_SAMPLER_PID:-}" ]]; then
        kill "$TOPAX_GPU_SAMPLER_PID" 2>/dev/null || true
        wait "$TOPAX_GPU_SAMPLER_PID" 2>/dev/null || true
        unset TOPAX_GPU_SAMPLER_PID
    fi
}

topax_write_memory_csv() (
    export LC_ALL=C

    local output="$1"
    local runtime_csv="$2"
    local time_file="$3"
    local gpu_samples="${4:-}"
    local gpu_baseline_mib="${5:-}"
    local iterations
    local peak_rss_kib
    local host_peak_gib
    local gpu_peak_mib=""
    local gpu_baseline_gib=""
    local gpu_peak_gib=""
    local gpu_delta_gib=""

    [[ -s "$runtime_csv" ]] || topax_die "runtime CSV is empty: $runtime_csv"
    iterations="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$runtime_csv")"
    (( iterations > 0 )) || topax_die "runtime CSV contains no iterations"

    peak_rss_kib="$(
        awk -F: \
            '/Maximum resident set size \(kbytes\)/ {
                gsub(/[[:space:]]/, "", $2); print $2; exit
            }' "$time_file"
    )"
    [[ "$peak_rss_kib" =~ ^[0-9]+$ ]] ||
        topax_die "could not read peak resident memory from $time_file"
    host_peak_gib="$(
        awk -v value="$peak_rss_kib" 'BEGIN { printf "%.12g", value / 1048576 }'
    )"

    if [[ -n "$gpu_samples" ]]; then
        [[ -s "$gpu_samples" ]] || topax_die "GPU sample file is empty"
        [[ "$gpu_baseline_mib" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
            topax_die "GPU baseline is unavailable"
        gpu_peak_mib="$(
            awk -F, '
                NR > 1 && $2 ~ /^[0-9]+([.][0-9]+)?$/ {
                    if (!seen || $2 > peak) peak = $2
                    seen = 1
                }
                END { if (seen) print peak }
            ' "$gpu_samples"
        )"
        [[ "$gpu_peak_mib" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
            topax_die "GPU samples contain no memory measurements"
        gpu_baseline_gib="$(
            awk -v value="$gpu_baseline_mib" \
                'BEGIN { printf "%.12g", value / 1024 }'
        )"
        gpu_peak_gib="$(
            awk -v value="$gpu_peak_mib" \
                'BEGIN { printf "%.12g", value / 1024 }'
        )"
        gpu_delta_gib="$(
            awk -v peak="$gpu_peak_mib" -v baseline="$gpu_baseline_mib" \
                'BEGIN { printf "%.12g", (peak - baseline) / 1024 }'
        )"
    fi

    {
        echo "design_iterations_completed,host_peak_rss_gib,gpu_baseline_gib,gpu_peak_gib,gpu_peak_above_baseline_gib"
        printf '%s,%s,%s,%s,%s\n' \
            "$iterations" "$host_peak_gib" "$gpu_baseline_gib" \
            "$gpu_peak_gib" "$gpu_delta_gib"
    } > "$output"
)
