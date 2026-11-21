#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
BENCH_DIR="${OPENMP_BENCH_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TOPAX_ROOT="${TOPAX_ROOT:-$(cd "$BENCH_DIR/../../.." && pwd)}"
OPENMP_SOURCE="$BENCH_DIR/TopOpt-in-OpenMP-GPU"
PATCH_FILE="$BENCH_DIR/patches/pcg_positivity_trace.patch"
SOURCE_REVISION="b46ed453ad5a05818b563e540ce176c9d85c6fe2"

die() {
    echo "ERROR: $*" >&2
    exit 2
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

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ -n "${ACCOUNT:-}" ]] || die "set ACCOUNT"
    [[ -n "${COMPILER_MODULE:-}" ]] || die "set COMPILER_MODULE"
    [[ -n "${SUITESPARSE_ROOT:-}" ]] || die "set SUITESPARSE_ROOT"
    [[ -n "${OPENBLAS_ROOT:-}" ]] || die "set OPENBLAS_ROOT"
    [[ -n "${GCC_RUNTIME_ROOT:-}" ]] || die "set GCC_RUNTIME_ROOT"
    [[ -d "$OPENMP_SOURCE/.git" ]] || die "run setup.sh first"
    [[ "$(git -C "$OPENMP_SOURCE" rev-parse HEAD)" == "$SOURCE_REVISION" ]] ||
        die "OpenMP checkout is not at the pinned revision"
    git -C "$OPENMP_SOURCE" apply --check "$PATCH_FILE" >/dev/null ||
        die "diagnostic patch does not apply to the pinned source"

    RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
    [[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] ||
        die "RUN_ID must use the UTC format YYYYMMDDTHHMMSSZ"
    PARTITION="${PARTITION:-gpu}"
    CPUS="${CPUS:-12}"
    MEMORY="${MEMORY:-64G}"
    WALLTIME="${WALLTIME:-00:30:00}"
    LOG_DIR="$BENCH_DIR/.slurm/$RUN_ID"
    RESULT_DIR="$TOPAX_ROOT/benchmarks/results/traeff_2023/openmp/diagnostics/pcg_positivity/$RUN_ID"
    mkdir -p "$LOG_DIR" "$RESULT_DIR"

    JOB_ID="$(
        sbatch --parsable \
            -A "$ACCOUNT" -p "$PARTITION" -N 1 --ntasks=1 \
            --cpus-per-task="$CPUS" --mem="$MEMORY" --gpus=1 \
            -t "$WALLTIME" -J "openmp-pcg-diagnostic" \
            -o "$LOG_DIR/pcg-diagnostic-%j.out" \
            --export=ALL,OPENMP_DIAGNOSTIC_WORKER=1,OPENMP_RUN_ID="$RUN_ID",OPENMP_BENCH_DIR="$BENCH_DIR",TOPAX_ROOT="$TOPAX_ROOT",COMPILER_MODULE="$COMPILER_MODULE",SUITESPARSE_ROOT="$SUITESPARSE_ROOT",OPENBLAS_ROOT="$OPENBLAS_ROOT",GCC_RUNTIME_ROOT="$GCC_RUNTIME_ROOT" \
            "$SCRIPT_PATH"
    )"
    echo "Run ID: $RUN_ID"
    echo "Submitted PCG diagnostic: $JOB_ID"
    exit 0
fi

[[ "${OPENMP_DIAGNOSTIC_WORKER:-}" == 1 ]] || die "submit this script from a login node"
[[ "$(uname -m)" == aarch64 ]] || die "diagnostic worker must run on AArch64"
RUN_ID="${OPENMP_RUN_ID:?}"
RESULT_DIR="$TOPAX_ROOT/benchmarks/results/traeff_2023/openmp/diagnostics/pcg_positivity/$RUN_ID"
SCRATCH="${SLURM_TMPDIR:-/tmp}/openmp-pcg-diagnostic-${SLURM_JOB_ID}"
SOURCE_DIR="$SCRATCH/source"
mkdir -p "$SOURCE_DIR" "$RESULT_DIR"

cleanup() {
    # shellcheck disable=SC2317
    rm -rf "$SCRATCH"
}
trap cleanup EXIT INT TERM

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

git -C "$OPENMP_SOURCE" archive "$SOURCE_REVISION" | tar -x -C "$SOURCE_DIR"
git -C "$SOURCE_DIR" apply "$PATCH_FILE"

BUILD_LOG="$RESULT_DIR/build.log"
make -C "$SOURCE_DIR" clean > "$BUILD_LOG" 2>&1
make -C "$SOURCE_DIR" -j "$OMP_NUM_THREADS" \
    CC=nvc \
    CFLAGS="-Wall -g -mp=gpu -gpu=cc90 -gpu=mem:separate -mp=noautopar -Msafeptr -O4" \
    DEFS="-DDTU_HPC" \
    INCL="-I$SUITESPARSE_ROOT/include/suitesparse -I$OPENBLAS_ROOT/include" \
    LIBS="-L$SUITESPARSE_LIBDIR -lcholmod -lsuitesparseconfig -L$OPENBLAS_LIBDIR -lopenblas -lm" \
    >> "$BUILD_LOG" 2>&1

COMPILER_VERSION="$(nvc --version | awk 'NF {print; exit}')"
cat > "$RESULT_DIR/configuration.toml" <<EOF
schema_version = 2
benchmark = "traeff_2023"
implementation = "openmp"
implementation_version = "0.9.1"
source_revision = "$SOURCE_REVISION"
diagnostic_patch = "pcg_positivity_trace.patch"
purpose = "pcg_preconditioner_positivity"
backend = "openmp_target_cuda"
elements = [128, 64, 64]
levels = 4
physics_tolerance = 1e-5
expected_failure_exit_status = 86
compiler = "$COMPILER_VERSION"
compiler_module = "$COMPILER_MODULE"
cpu_threads = ${SLURM_CPUS_PER_TASK:-12}
EOF

TRACE="$RESULT_DIR/trace.log"
set +e
TOP3D_ABORT_ON_NONPOSITIVE_RZ=1 \
    /usr/bin/time -v stdbuf -oL -eL \
    "$SOURCE_DIR/top3d" \
        -x 16 -y 8 -z 8 -l 4 -f 0.10 -r 1.5 -i 20 -v 1 -w 0 \
        > "$TRACE" 2>&1
APPLICATION_STATUS=$?
set -e

awk '
BEGIN {OFS=","; print "solve,pcg_iteration,r_transpose_z"}
/^PCG_PRECONDITIONER / {
    split($2, solve, "="); split($3, iteration, "="); split($4, rz, "=");
    print solve[2], iteration[2], rz[2]
}
' "$TRACE" > "$RESULT_DIR/pcg_preconditioner_trace.csv"

awk -v status="$APPLICATION_STATUS" '
BEGIN {OFS=","; print "outcome,application_exit_status,solve,pcg_iteration,r_transpose_z"; found=0}
/^PCG_POSITIVITY_FAILURE / && !found {
    split($2, solve, "="); split($3, iteration, "="); split($4, rz, "=");
    print "nonpositive_rz_reproduced", status, solve[2], iteration[2], rz[2];
    found=1
}
END {if (!found) print "nonpositive_rz_not_observed", status, "", "", ""}
' "$TRACE" > "$RESULT_DIR/summary.csv"

cat "$RESULT_DIR/summary.csv"
if grep -q '^PCG_POSITIVITY_FAILURE ' "$TRACE" && (( APPLICATION_STATUS == 86 )); then
    exit 0
fi
die "expected non-positive r-transpose-z value was not reproduced"

