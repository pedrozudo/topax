#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
BENCH_DIR="${TOP3DXL_BENCH_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$BENCH_DIR/../../.." && pwd)}"
TOP3DXL_SOURCE="$BENCH_DIR/TOP3D_XL"
PATCH_FILE="$BENCH_DIR/patches/headless_benchmark.patch"
SOURCE_REVISION="46a631fb631da1673de72c0ec17a318b0a2a4e41"

RESOLUTIONS=(96 192 384 768 800)
MESH_X=(96 192 384 768 800)
MESH_Y=(48 96 192 384 400)
MESH_Z=(48 96 192 384 400)

die() {
    echo "ERROR: $*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
Usage: wang_cpu.sh [options]

Options:
  --run-id ID       UTC ID in YYYYMMDDTHHMMSSZ format (default: now)
  --task-index N    submit only mesh index N, from 0 through 4
EOF
}

acquire_result_lock() {
    local output_dir="$1"
    local lock="$output_dir/.run.lock"

    mkdir -p "$output_dir"
    [[ ! -e "$output_dir/runtime.csv" ]] || die "result already exists: $output_dir/runtime.csv"
    [[ ! -e "$output_dir/memory.csv" ]] || die "result already exists: $output_dir/memory.csv"
    mkdir "$lock" 2>/dev/null || die "another measurement is using $output_dir"
    printf '%s\n' "$lock"
}

hierarchy_levels() {
    local voxels=$((EX * EY * EZ))
    local count=0
    while (( voxels >= 50000 )); do
        count=$((count + 1))
        voxels=$(((voxels + 4) / 8))
    done
    (( count >= 3 )) || count=3
    printf '%s\n' "$count"
}

parse_results() {
    local raw="$1"
    local time_file="$2"
    local runtime_output="$3"
    local memory_output="$4"

    python3 - "$raw" "$time_file" "$runtime_output" "$memory_output" <<'PY'
import csv
import re
import sys
from pathlib import Path

raw_path, time_path, runtime_path, memory_path = map(Path, sys.argv[1:])
text = raw_path.read_text(errors="replace")
time_text = time_path.read_text(errors="replace")
number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

cg_pattern = re.compile(
    rf"CG solver converged at iteration\s+(\d+)\s+"
    rf"to a solution with relative residual\s+({number})"
)
objective_pattern = re.compile(
    rf"It\.:\s*(\d+)\s+Obj\.:\s*({number})\s+Vol\.:\s*({number})\s+"
    rf"Sharp\.:\s*({number})\s+Cons\.:\s*({number})\s+Ch\.:\s*({number})"
)
time_pattern = re.compile(
    rf"It\.:\s*(\d+)\s+\(Time\)\.\.\.\s+Total per-It\.:\s*({number})s;\s*"
    rf"Assemb\.:\s*({number})s;\s*CG:\s*({number})s;\s*"
    rf"Opti\.:\s*({number})s;\s*Filtering:\s*({number})s\."
)

events = []
for match in cg_pattern.finditer(text):
    events.append((match.start(), "cg", match))
for match in objective_pattern.finditer(text):
    events.append((match.start(), "objective", match))
events.sort(key=lambda event: event[0])

rows = {}
pending_cg = None
for _, kind, match in events:
    if kind == "cg":
        pending_cg = (int(match.group(1)), float(match.group(2)))
        continue
    iteration = int(match.group(1))
    row = rows.setdefault(iteration, {})
    row.update(
        compliance=float(match.group(2)),
        density=float(match.group(3)),
        non_discreteness=float(match.group(4)),
        volume_constraint=float(match.group(5)),
        change=float(match.group(6)),
    )
    if pending_cg is not None:
        row["physics_cg_iterations"] = pending_cg[0]
        row["physics_relative_residual"] = pending_cg[1]
    pending_cg = None

for match in time_pattern.finditer(text):
    iteration = int(match.group(1))
    rows.setdefault(iteration, {})["iteration_time_s"] = float(match.group(2))

expected = list(range(1, 51))
if sorted(rows) != expected:
    raise SystemExit(f"expected design iterations 1 through 50, found {sorted(rows)}")
required = {
    "compliance",
    "density",
    "non_discreteness",
    "volume_constraint",
    "change",
    "physics_cg_iterations",
    "physics_relative_residual",
    "iteration_time_s",
}
for iteration, row in rows.items():
    missing = required - row.keys()
    if missing:
        raise SystemExit(f"iteration {iteration} is missing: {sorted(missing)}")

runtime_fields = [
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
with runtime_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=runtime_fields)
    writer.writeheader()
    for iteration in expected:
        row = rows[iteration]
        writer.writerow(
            {
                "iteration": iteration,
                "compliance": row["compliance"],
                "mean_design_density": row["density"],
                "mean_physical_density": "",
                "volume_constraint": row["volume_constraint"],
                "non_discreteness_reported": row["non_discreteness"],
                "change": row["change"],
                "physics_cg_iterations": row["physics_cg_iterations"],
                "physics_relative_residual": row["physics_relative_residual"],
                "filter_cg_iterations": "",
                "filter_relative_residual": "",
                "iteration_time_s": row["iteration_time_s"],
            }
        )

rss_match = re.search(
    r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text
)
if not rss_match:
    raise SystemExit("GNU time did not report maximum resident memory")
peak_rss_gib = int(rss_match.group(1)) / 1024**2
memory_fields = [
    "design_iterations_completed",
    "host_peak_rss_gib",
    "gpu_baseline_gib",
    "gpu_peak_gib",
    "gpu_peak_above_baseline_gib",
]
with memory_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=memory_fields)
    writer.writeheader()
    writer.writerow(
        {
            "design_iterations_completed": 50,
            "host_peak_rss_gib": peak_rss_gib,
            "gpu_baseline_gib": "",
            "gpu_peak_gib": "",
            "gpu_peak_above_baseline_gib": "",
        }
    )
PY
}

write_configuration() {
    local output_dir="$1"
    local matlab_version="$2"
    local levels
    levels="$(hierarchy_levels)"

    cat > "$output_dir/.configuration.$$.toml" <<EOF
schema_version = 2
benchmark = "wang_2025"
implementation = "top3d_xl"
source_revision = "$SOURCE_REVISION"
headless_patch = "headless_benchmark.patch"
backend = "cpu"
elements = [$EX, $EY, $EZ]
resolution = "$RESOLUTION"
levels = $levels
hierarchy_selection = "adaptive"
coarsest_element_target = 50000
first_coarsening_factor = 4
subsequent_coarsening_factor = 2
cycle = "adapted V-cycle"
jacobi_damping = 0.6
coarse_solver = "MATLAB sparse Cholesky"
precision = "float64"
physics_tolerance = 1e-3
physics_max_iterations = 800
filter = "PDE sensitivity filter"
filter_radius_elements = 1.7320508075688772
filter_tolerance = 1e-4
volume_fraction = 0.12
simp_penalty = 3
minimum_relative_stiffness = 1e-6
optimality_criteria_move_limit = 0.2
matlab_version = "$matlab_version"
cpu_threads = ${SLURM_CPUS_PER_TASK:-32}

[measurements.runtime]
design_iteration_limit = 50
full_benchmark = true

[measurements.memory]
design_iteration_limit = 50
full_benchmark = true
host_metric = "process peak RSS"
EOF
    mv "$output_dir/.configuration.$$.toml" "$output_dir/configuration.toml"
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
        [[ "$TASK_INDEX" =~ ^[0-4]$ ]] || die "--task-index must be an integer from 0 through 4"
    fi
    [[ -n "${ACCOUNT:-}" ]] || die "set ACCOUNT"
    [[ -n "${MATLAB_MODULE:-}" ]] || die "set MATLAB_MODULE"
    [[ -d "$TOP3DXL_SOURCE/.git" ]] || die "run setup.sh before submitting"
    [[ "$(git -C "$TOP3DXL_SOURCE" rev-parse HEAD)" == "$SOURCE_REVISION" ]] ||
        die "TOP3D_XL checkout is not at the pinned revision"
    git -C "$TOP3DXL_SOURCE" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1 ||
        die "TOP3D_XL headless benchmark patch is missing"

    RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
    [[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] ||
        die "run ID must use the UTC format YYYYMMDDTHHMMSSZ"
    PARTITION="${PARTITION:-cpu}"
    CPUS="${CPUS:-32}"
    SHORT_WALLTIME="${SHORT_WALLTIME:-04:00:00}"
    LONG_WALLTIME="${LONG_WALLTIME:-36:00:00}"
    LOG_DIR="$BENCH_DIR/.slurm/$RUN_ID"
    mkdir -p "$LOG_DIR"

    echo "Run ID: $RUN_ID"

    submit_array() {
        local array_spec="$1"
        local walltime="$2"
        local label="$3"
        local job_id
        job_id="$(
            sbatch --parsable \
                -A "$ACCOUNT" -p "$PARTITION" -N 1 --ntasks=1 \
                --cpus-per-task="$CPUS" -t "$walltime" \
                --array="$array_spec" -J "top3dxl-wang-cpu" \
                -o "$LOG_DIR/wang-cpu-%A_%a.out" \
                --export=ALL,TOP3DXL_RUN_ID="$RUN_ID",TOP3DXL_BENCH_DIR="$BENCH_DIR",TOPAX_ROOT="$TOPAX_ROOT",MATLAB_MODULE="$MATLAB_MODULE" \
                "$SCRIPT_PATH"
        )"
        echo "Submitted $label array: $job_id"
    }

    if [[ -n "$TASK_INDEX" ]]; then
        if (( TASK_INDEX <= 2 )); then
            submit_array "$TASK_INDEX" "$SHORT_WALLTIME" "short"
        else
            submit_array "$TASK_INDEX" "$LONG_WALLTIME" "long"
        fi
    else
        submit_array "0-2" "$SHORT_WALLTIME" "short"
        submit_array "3-4" "$LONG_WALLTIME" "long"
    fi
    exit 0
fi

RUN_ID="${TOP3DXL_RUN_ID:?}"
INDEX="${SLURM_ARRAY_TASK_ID:?}"
RESOLUTION="${RESOLUTIONS[$INDEX]}"
EX="${MESH_X[$INDEX]}"
EY="${MESH_Y[$INDEX]}"
EZ="${MESH_Z[$INDEX]}"
MESH="${EX}x${EY}x${EZ}"
OUTPUT_DIR="$TOPAX_ROOT/benchmarks/results/wang_2025/top3d_xl/cpu/$MESH/$RUN_ID"
LOCK="$(acquire_result_lock "$OUTPUT_DIR")"
SCRATCH="${SLURM_TMPDIR:-/tmp}/top3dxl-wang-${SLURM_JOB_ID}-${INDEX}"
mkdir -p "$SCRATCH"

cleanup() {
    rm -f "$OUTPUT_DIR/.runtime.$$.csv" "$OUTPUT_DIR/.memory.$$.csv" \
        "$OUTPUT_DIR/.configuration.$$.toml"
    rm -rf "$LOCK" "$SCRATCH"
}
trap cleanup EXIT INT TERM

module --force purge
module load "$MATLAB_MODULE"
command -v matlab >/dev/null || die "MATLAB executable is unavailable"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export LC_ALL=C

cp "$TOP3DXL_SOURCE/TOP3D_XL.m" "$SCRATCH/TOP3D_XL.m"
RAW="$SCRATCH/output.txt"
TIME_FILE="$SCRATCH/time.txt"
TEMP_RUNTIME="$OUTPUT_DIR/.runtime.$$.csv"
TEMP_MEMORY="$OUTPUT_DIR/.memory.$$.csv"

cd "$SCRATCH"
set +e
/usr/bin/time -v -o "$TIME_FILE" \
    matlab -batch "fprintf('TOP3DXL_MATLAB_VERSION=%s\\n',version); set(groot,'defaultFigureVisible','off'); TOP3D_XL(true($EY,$EX,$EZ),'GLOBAL',0.12,50,sqrt(3));" \
    > "$RAW" 2>&1
STATUS=$?
set -e
if (( STATUS != 0 )); then
    cat "$RAW" >&2
    exit "$STATUS"
fi

parse_results "$RAW" "$TIME_FILE" "$TEMP_RUNTIME" "$TEMP_MEMORY"
MATLAB_VERSION="$(sed -n 's/^TOP3DXL_MATLAB_VERSION=//p' "$RAW" | head -n 1)"
[[ -n "$MATLAB_VERSION" ]] || die "MATLAB version marker is missing"
MATLAB_VERSION="${MATLAB_VERSION//\"/\\\"}"
write_configuration "$OUTPUT_DIR" "$MATLAB_VERSION"
mv "$TEMP_RUNTIME" "$OUTPUT_DIR/runtime.csv"
mv "$TEMP_MEMORY" "$OUTPUT_DIR/memory.csv"
echo "Wrote $OUTPUT_DIR/{runtime,memory}.csv"

