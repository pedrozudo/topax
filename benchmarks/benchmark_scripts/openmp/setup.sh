#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/TopOpt-in-OpenMP-GPU"
UPSTREAM_URL="https://github.com/topopt/TopOpt-in-OpenMP-GPU.git"
UPSTREAM_REVISION="b46ed453ad5a05818b563e540ce176c9d85c6fe2"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

command -v git >/dev/null || die "git is required"

if [[ ! -e "$SOURCE_DIR" ]]; then
    git clone "$UPSTREAM_URL" "$SOURCE_DIR"
elif [[ ! -d "$SOURCE_DIR/.git" ]]; then
    die "$SOURCE_DIR exists but is not a Git checkout"
fi

CURRENT_REVISION="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$CURRENT_REVISION" != "$UPSTREAM_REVISION" ]]; then
    if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
        die "OpenMP checkout has local changes at the wrong revision"
    fi
    git -C "$SOURCE_DIR" fetch origin "$UPSTREAM_REVISION"
    git -C "$SOURCE_DIR" checkout --detach "$UPSTREAM_REVISION"
fi
git -C "$SOURCE_DIR" diff --quiet ||
    die "OpenMP checkout contains tracked source modifications"

echo "OpenMP source ready at $SOURCE_DIR"
echo "Run build.sh on the target compute-node architecture."

