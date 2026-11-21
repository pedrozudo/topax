#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/TopOpt-in-OpenMP-GPU"
BUILD_DIR="$SCRIPT_DIR/build"
CC="${CC:-nvc}"
GPU_ARCH="${GPU_ARCH:-cc90}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-12}}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "${SUITESPARSE_ROOT:-}" ]] || die "set SUITESPARSE_ROOT"
[[ -n "${OPENBLAS_ROOT:-}" ]] || die "set OPENBLAS_ROOT"
[[ -n "${GCC_RUNTIME_ROOT:-}" ]] || die "set GCC_RUNTIME_ROOT"
command -v "$CC" >/dev/null || die "compiler not found: $CC"
[[ -f "$SOURCE_DIR/Makefile" ]] || die "run setup.sh before build.sh"
[[ "$(uname -m)" == aarch64 ]] || die "build.sh must run on the AArch64 target compute node"
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "b46ed453ad5a05818b563e540ce176c9d85c6fe2" ]] ||
    die "OpenMP checkout is not at the pinned revision"
git -C "$SOURCE_DIR" diff --quiet || die "OpenMP checkout contains tracked source modifications"

if [[ -d "$SUITESPARSE_ROOT/lib64" ]]; then
    SUITESPARSE_LIBDIR="$SUITESPARSE_ROOT/lib64"
elif [[ -d "$SUITESPARSE_ROOT/lib" ]]; then
    SUITESPARSE_LIBDIR="$SUITESPARSE_ROOT/lib"
else
    die "SuiteSparse library directory not found below $SUITESPARSE_ROOT"
fi
if [[ -d "$OPENBLAS_ROOT/lib64" ]]; then
    OPENBLAS_LIBDIR="$OPENBLAS_ROOT/lib64"
elif [[ -d "$OPENBLAS_ROOT/lib" ]]; then
    OPENBLAS_LIBDIR="$OPENBLAS_ROOT/lib"
else
    die "OpenBLAS library directory not found below $OPENBLAS_ROOT"
fi
[[ -f "$SUITESPARSE_ROOT/include/suitesparse/cholmod.h" ]] ||
    die "CHOLMOD header not found below $SUITESPARSE_ROOT"
[[ -f "$OPENBLAS_ROOT/include/cblas.h" ]] ||
    die "CBLAS header not found below $OPENBLAS_ROOT"
[[ -d "$GCC_RUNTIME_ROOT" ]] || die "GCC runtime directory does not exist"

export LD_LIBRARY_PATH="$SUITESPARSE_LIBDIR:$OPENBLAS_LIBDIR:$GCC_RUNTIME_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CFLAGS="${CFLAGS:--Wall -g -mp=gpu -gpu=$GPU_ARCH -gpu=mem:separate -mp=noautopar -Msafeptr -O4}"

make -C "$SOURCE_DIR" clean
make -C "$SOURCE_DIR" -j "$BUILD_JOBS" \
    CC="$CC" \
    CFLAGS="$CFLAGS" \
    DEFS="-DDTU_HPC" \
    INCL="-I$SUITESPARSE_ROOT/include/suitesparse -I$OPENBLAS_ROOT/include" \
    LIBS="-L$SUITESPARSE_LIBDIR -lcholmod -lsuitesparseconfig -L$OPENBLAS_LIBDIR -lopenblas -lm"

mkdir -p "$BUILD_DIR"
cp "$SOURCE_DIR/top3d" "$BUILD_DIR/top3d"
"$CC" --version > "$BUILD_DIR/compiler-version.txt" 2>&1
cat > "$BUILD_DIR/build.toml" <<EOF
source_revision = "$(git -C "$SOURCE_DIR" rev-parse HEAD)"
architecture = "$(uname -m)"
compiler = "$CC"
gpu_architecture = "$GPU_ARCH"
memory_mode = "separate"
EOF

file "$BUILD_DIR/top3d"
ldd "$BUILD_DIR/top3d"
echo "Built $BUILD_DIR/top3d for $(uname -m)"

