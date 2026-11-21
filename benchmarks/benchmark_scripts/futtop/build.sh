#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/Futtop"
BUILD_DIR="$SCRIPT_DIR/build"
CC="${CC:-cc}"
OPENCL_LIBS="${OPENCL_LIBS:--lOpenCL}"
CFLAGS="${CFLAGS:--O3 -march=native}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

command -v "$CC" >/dev/null || die "C compiler not found: $CC"
[[ -f "$SOURCE_DIR/libmultigrid.c" ]] || die "run setup.sh before build.sh"
[[ -f "$SOURCE_DIR/libmultigrid.h" ]] || die "run setup.sh before build.sh"
mkdir -p "$BUILD_DIR"

# CPPFLAGS, LDFLAGS, and OPENCL_LIBS may be supplied by the build environment.
# Intentional word splitting permits conventional compiler flag strings.
# shellcheck disable=SC2086
"$CC" -std=c99 -fopenmp $CFLAGS ${CPPFLAGS:-} -I"$SOURCE_DIR" \
    -c "$SOURCE_DIR/libmultigrid.c" -o "$BUILD_DIR/libmultigrid.o"
# shellcheck disable=SC2086
"$CC" -std=c99 -fopenmp $CFLAGS ${CPPFLAGS:-} -I"$SOURCE_DIR" \
    -c "$SOURCE_DIR/io.c" -o "$BUILD_DIR/io.o"
# shellcheck disable=SC2086
"$CC" -std=c99 -fopenmp $CFLAGS ${CPPFLAGS:-} -I"$SOURCE_DIR" \
    -c "$SOURCE_DIR/futtop.c" -o "$BUILD_DIR/futtop.o"
# shellcheck disable=SC2086
"$CC" -fopenmp $CFLAGS ${LDFLAGS:-} \
    "$BUILD_DIR/futtop.o" "$BUILD_DIR/libmultigrid.o" "$BUILD_DIR/io.o" \
    -lm $OPENCL_LIBS -o "$BUILD_DIR/futtop"

"$BUILD_DIR/futtop" --help >/dev/null 2>&1 || true
"$CC" --version | head -n 1 > "$BUILD_DIR/compiler-version.txt"
echo "Built $BUILD_DIR/futtop for $(uname -m)"

