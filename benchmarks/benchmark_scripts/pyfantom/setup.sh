#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/pyFANTOM"
VENV_DIR="$SCRIPT_DIR/.venv"
PATCH_FILE="$SCRIPT_DIR/patches/record_pcg_state.patch"
UPSTREAM_URL="https://github.com/ahnobari/pyFANTOM.git"
UPSTREAM_REVISION="ffa4c95f087f86933629cb1926030b80501c3136"
PYTHON="${PYTHON:-python3.9}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

command -v git >/dev/null || die "git is required"
command -v "$PYTHON" >/dev/null || die "Python interpreter not found: $PYTHON"

if [[ -n "${CUDA_HOME:-}" ]]; then
    [[ -d "$CUDA_HOME" ]] || die "CUDA_HOME does not exist: $CUDA_HOME"
    CUDA_LIBDIR=""
    for candidate in \
        "$CUDA_HOME/targets/sbsa-linux/lib" \
        "$CUDA_HOME/targets/x86_64-linux/lib" \
        "$CUDA_HOME/lib64"; do
        if [[ -e "$candidate/libnvrtc.so.13" ]]; then
            CUDA_LIBDIR="$candidate"
            break
        fi
    done
    [[ -n "$CUDA_LIBDIR" ]] || die "libnvrtc.so.13 was not found below CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

"$PYTHON" -c 'import ctypes; ctypes.CDLL("libnvrtc.so.13")' >/dev/null 2>&1 ||
    die "libnvrtc.so.13 is unavailable; set CUDA_HOME or load the CUDA 13 runtime environment"

if [[ -n "${CHOLMOD_ROOT:-}" ]]; then
    [[ -d "$CHOLMOD_ROOT" ]] || die "CHOLMOD_ROOT does not exist: $CHOLMOD_ROOT"
    if [[ -d "$CHOLMOD_ROOT/lib64" ]]; then
        CHOLMOD_LIBDIR="$CHOLMOD_ROOT/lib64"
    elif [[ -d "$CHOLMOD_ROOT/lib" ]]; then
        CHOLMOD_LIBDIR="$CHOLMOD_ROOT/lib"
    else
        die "CHOLMOD_ROOT contains neither lib64 nor lib"
    fi
    export CPATH="$CHOLMOD_ROOT/include/suitesparse:$CHOLMOD_ROOT/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$CHOLMOD_LIBDIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$CHOLMOD_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ! -e "$SOURCE_DIR" ]]; then
    git clone "$UPSTREAM_URL" "$SOURCE_DIR"
elif [[ ! -d "$SOURCE_DIR/.git" ]]; then
    die "$SOURCE_DIR exists but is not a Git checkout"
fi

CURRENT_REVISION="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$CURRENT_REVISION" != "$UPSTREAM_REVISION" ]]; then
    if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
        die "pyFANTOM checkout has local changes at the wrong revision"
    fi
    git -C "$SOURCE_DIR" fetch origin "$UPSTREAM_REVISION"
    git -C "$SOURCE_DIR" checkout --detach "$UPSTREAM_REVISION"
fi

if git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    :
elif git -C "$SOURCE_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
    git -C "$SOURCE_DIR" apply "$PATCH_FILE"
else
    die "cannot apply the pyFANTOM PCG instrumentation patch"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
[[ -x "$PYTHON_BIN" ]] || die "virtual environment is incomplete: $VENV_DIR"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel cython
"$PYTHON_BIN" -m pip install \
    numpy==2.0.2 \
    scipy==1.13.1 \
    numba==0.60.0
"$PYTHON_BIN" -m pip install \
    --no-binary=scikit-sparse \
    scikit-sparse==0.4.16
"$PYTHON_BIN" -m pip install \
    cupy-cuda13x==13.6.0 \
    matplotlib==3.9.4 \
    vtk==9.6.2 \
    k3d==3.2.0 \
    sympy==1.14.0
"$PYTHON_BIN" -m pip install --no-deps --editable "$SOURCE_DIR"

"$PYTHON_BIN" - <<'PY'
import importlib.metadata

import cupy
import numpy
import scipy
from pyFANTOM.CUDA import MultiGrid
from sksparse.cholmod import cholesky

print(f"pyFANTOM {importlib.metadata.version('pyFANTOM')}")
print(f"CuPy {cupy.__version__}")
print(f"NumPy {numpy.__version__}")
print(f"SciPy {scipy.__version__}")
print(f"scikit-sparse {importlib.metadata.version('scikit-sparse')}")
print(f"MultiGrid: {MultiGrid.__module__}")
print(f"CHOLMOD: {cholesky.__module__}")
PY

echo "pyFANTOM setup complete: $VENV_DIR"
