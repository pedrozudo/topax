# TOP3D_XL benchmarks

This directory contains the setup and submission scripts used to run the Wang et al. three-dimensional cantilever benchmark with [TOP3D_XL](https://github.com/PSLer/TOP3D_XL). The external source is checked out at commit `46a631fb631da1673de72c0ec17a318b0a2a4e41` and is not tracked by Git.

## Prerequisites

The setup requires Git. Running the benchmark requires MATLAB and Python 3. MATLAB is treated as an external prerequisite and selected through an environment-module name passed to the submission script.

## Setup

Clone the pinned source and apply the tracked headless-output patch:

```bash
bash setup.sh
```

The patch returns after the final design iteration and therefore skips volume export and visualization. It does not modify setup, optimization, finite-element analysis, filtering, or the multigrid solver.

## Run

Submit one small case as a smoke test:

```bash
ACCOUNT=your-slurm-account \
MATLAB_MODULE=matlab/R2025b \
bash slurm/wang_cpu.sh --task-index 0
```

Submit all five meshes, from $96\times48\times48$ through $800\times400\times400$, by omitting `--task-index 0`. Set `PARTITION` when the CPU partition is not named `cpu`. The launcher submits R96--R384 with a four-hour limit and R768--R800 with a 36-hour limit under the same run ID.

## Results

Each mesh writes:

```text
benchmarks/results/wang_2025/top3d_xl/cpu/<mesh>/<run-id>/
├── configuration.toml
├── runtime.csv
└── memory.csv
```

Runtime contains the 50 design iterations reported by TOP3D_XL. Memory is the natural process peak RSS from the same execution.

