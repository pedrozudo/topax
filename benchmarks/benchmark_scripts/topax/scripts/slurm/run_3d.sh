#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: run_3d.sh

Submit the Träff GPU and Wang CPU/GPU benchmark configurations.
Each configuration receives its own timestamped run ID.
EOF
    exit 0
fi
if (( $# )); then
    echo "ERROR: run_3d.sh does not accept arguments" >&2
    exit 2
fi

echo "Submitting Träff GPU L4-C2"
"$SCRIPT_DIR/traeff_gpu.sh" --levels 4 --crossover-level 2

echo
echo "Submitting Träff GPU L4-C3"
"$SCRIPT_DIR/traeff_gpu.sh" --levels 4 --crossover-level 3

echo
echo "Submitting Wang CPU L4-C3"
"$SCRIPT_DIR/wang_cpu.sh" --levels 4 --crossover-level 3

echo
echo "Submitting Wang CPU L5-C4"
"$SCRIPT_DIR/wang_cpu.sh" --levels 5 --crossover-level 4

echo
echo "Submitting Wang GPU L4-C3"
"$SCRIPT_DIR/wang_gpu.sh" \
    --levels 4 --crossover-level 3 --max-resolution 800

echo
echo "Submitting Wang GPU L5-C4"
"$SCRIPT_DIR/wang_gpu.sh" --levels 5 --crossover-level 4
