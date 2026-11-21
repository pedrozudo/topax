#!/usr/bin/env bash
# shellcheck disable=SC2034
set -euo pipefail

TOPAX_LOCAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$TOPAX_LOCAL_SCRIPT_DIR/../../../.." && pwd)}"
TOPAX_LOCAL_LAUNCHER="${BASH_SOURCE[0]}"
TOPAX_LOCAL_TITLE="Sigmund 2007"
TOPAX_LOCAL_BENCHMARK="sigmund_2007"
TOPAX_LOCAL_PROGRAM="$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/sigmund_2007/mbb_conv_filter.py"
TOPAX_LOCAL_RUN_ROOT="$TOPAX_LOCAL_SCRIPT_DIR/.runs/sigmund_2007"
TOPAX_LOCAL_CASES=(120x40 360x120 750x250 1500x500 3000x1000 5400x1800)
TOPAX_LOCAL_MESH_X=(120 360 750 1500 3000 5400)
TOPAX_LOCAL_MESH_Y=(40 120 250 500 1000 1800)
TOPAX_LOCAL_LEVELS=(2 2 2 3 3 3)

topax_local_build_command() {
    local python_bin="$1"
    local case_name="$2"
    local levels="$3"
    local max_iterations="$4"
    local runtime_csv="$5"
    local artifact_dir="$6"
    TOPAX_LOCAL_COMMAND=(
        "$python_bin" -u "$TOPAX_LOCAL_PROGRAM"
        --case "$case_name"
        --device gpu
        --n-levels "$levels"
        --max-iterations "$max_iterations"
        --runtime-csv "$runtime_csv"
        --output-dir "$artifact_dir"
    )
}

topax_local_write_problem_configuration() {
    local output="$1"
    local ex="$2"
    local ey="$3"
    local levels="$4"
    local gpu_name="$5"
    local crossover=$((levels - 1))
    cat > "$output" <<EOF
schema_version = 2
benchmark = "sigmund_2007"
implementation = "topax"
backend = "gpu"
elements = [$ex, $ey]
levels = $levels
galerkin_crossover_level = $crossover
precision = "float64"
physics_tolerance = 1e-10
physics_max_iterations = 500
jacobi_damping = 0.8
optimization_method = "MMA"
filter = "linear density convolution"
filter_radius_elements = 3.5
volume_fraction = 0.5
simp_penalty = 3
minimum_relative_stiffness = 1e-9
gpu_name = "$gpu_name"
EOF
}

# shellcheck disable=SC1091
source "$TOPAX_LOCAL_SCRIPT_DIR/local_gpu_common.sh"
topax_run_local_gpu_campaign "$@"
