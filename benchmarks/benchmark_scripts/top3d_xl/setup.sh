#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/TOP3D_XL"
PATCH_FILE="$SCRIPT_DIR/patches/headless_benchmark.patch"
UPSTREAM_URL="https://github.com/PSLer/TOP3D_XL.git"
UPSTREAM_REVISION="46a631fb631da1673de72c0ec17a318b0a2a4e41"

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
        die "TOP3D_XL checkout has local changes at the wrong revision"
    fi
    git -C "$SOURCE_DIR" fetch origin "$UPSTREAM_REVISION"
    git -C "$SOURCE_DIR" checkout --detach "$UPSTREAM_REVISION"
fi

if git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    :
elif git -C "$SOURCE_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
    git -C "$SOURCE_DIR" apply "$PATCH_FILE"
else
    die "cannot apply the TOP3D_XL headless benchmark patch"
fi

echo "TOP3D_XL source ready at $SOURCE_DIR"

