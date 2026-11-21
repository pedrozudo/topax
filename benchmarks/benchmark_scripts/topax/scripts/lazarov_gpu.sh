#!/usr/bin/env bash
# shellcheck disable=SC2034
set -euo pipefail

TOPAX_LOCAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$TOPAX_LOCAL_SCRIPT_DIR/../../../.." && pwd)}"
TOPAX_LOCAL_LAUNCHER="${BASH_SOURCE[0]}"
TOPAX_LOCAL_TITLE="Lazarov et al. 2011"
TOPAX_LOCAL_BENCHMARK="lazarov_2011"
TOPAX_LOCAL_PROGRAM="$TOPAX_ROOT/benchmarks/benchmark_scripts/topax/lazarov_2011/cantilever_helmholtz_density.py"
TOPAX_LOCAL_RUN_ROOT="$TOPAX_LOCAL_SCRIPT_DIR/.runs/lazarov_2011"
TOPAX_LOCAL_CASES=(400x100 800x200 1200x300 1600x400 3200x800 6400x1600)
TOPAX_LOCAL_MESH_X=(400 800 1200 1600 3200 6400)
TOPAX_LOCAL_MESH_Y=(100 200 300 400 800 1600)
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
benchmark = "lazarov_2011"
implementation = "topax"
backend = "gpu"
elements = [$ex, $ey]
levels = $levels
galerkin_crossover_level = $crossover
precision = "float64"
physics_tolerance = 1e-10
physics_max_iterations = 500
jacobi_damping = 0.8
optimization_method = "OC"
filter = "Helmholtz PDE density filter"
paper_filter_radius = 0.02
topax_filter_radius = 0.0692820323027551
volume_fraction = 0.5
simp_penalty = 3
minimum_design_density = 1e-3
minimum_relative_stiffness = 1e-9
gpu_name = "$gpu_name"
EOF
}

# shellcheck disable=SC1091
source "$TOPAX_LOCAL_SCRIPT_DIR/local_gpu_common.sh"
topax_run_local_gpu_campaign "$@"
