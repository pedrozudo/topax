#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/Futtop"
BUILD_DIR="$SCRIPT_DIR/build"
PATCH_FILE="$SCRIPT_DIR/patches/disable_vtr_output.patch"
UPSTREAM_URL="https://github.com/topopt/futtop.git"
UPSTREAM_REVISION="c3cd2b00632ad57cd3d8917246552b2fb4e99e01"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "${FUTHARK_HOME:-}" ]] || die "set FUTHARK_HOME to the Futhark installation root"
FUTHARK_BIN="$FUTHARK_HOME/bin/futhark"
[[ -x "$FUTHARK_BIN" ]] || die "Futhark compiler not found: $FUTHARK_BIN"
command -v git >/dev/null || die "git is required"

if ! FUTHARK_VERSION_OUTPUT="$($FUTHARK_BIN --version 2>&1)"; then
    die "the Futhark compiler cannot run on $(uname -m); run setup.sh on a compatible host, then run build.sh on the target compute node"
fi
FUTHARK_VERSION="$(printf '%s\n' "$FUTHARK_VERSION_OUTPUT" | head -n 1)"
[[ "$FUTHARK_VERSION" == "Futhark 0.21.11" ]] ||
    die "Futhark 0.21.11 is required; found: $FUTHARK_VERSION"

if [[ ! -e "$SOURCE_DIR" ]]; then
    git clone "$UPSTREAM_URL" "$SOURCE_DIR"
elif [[ ! -d "$SOURCE_DIR/.git" ]]; then
    die "$SOURCE_DIR exists but is not a Git checkout"
fi

CURRENT_REVISION="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$CURRENT_REVISION" != "$UPSTREAM_REVISION" ]]; then
    if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
        die "Futtop checkout has local changes at the wrong revision"
    fi
    git -C "$SOURCE_DIR" fetch origin "$UPSTREAM_REVISION"
    git -C "$SOURCE_DIR" checkout --detach "$UPSTREAM_REVISION"
fi

if git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    :
elif git -C "$SOURCE_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
    git -C "$SOURCE_DIR" apply "$PATCH_FILE"
else
    die "cannot apply the Futtop output patch"
fi

(
    cd "$SOURCE_DIR"
    "$FUTHARK_BIN" opencl --library libmultigrid.fut
)

mkdir -p "$BUILD_DIR"
printf '%s\n' "$FUTHARK_VERSION_OUTPUT" > "$BUILD_DIR/futhark-version.txt"
echo "Generated the Futtop OpenCL C library in $SOURCE_DIR"
echo "Run build.sh on the target compute-node architecture before submitting benchmarks."

