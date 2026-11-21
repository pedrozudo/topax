# OpenMP implementation benchmarks

This directory contains the setup and submission scripts used to measure the memory consumption of the multiresolution OpenMP implementation accompanying [Träff et al.](https://github.com/topopt/TopOpt-in-OpenMP-GPU) v0.9.1. The external source is checked out at commit `b46ed453ad5a05818b563e540ce176c9d85c6fe2` and is not tracked by Git.

## Prerequisites

The setup requires Git. Building and running require an OpenMP target-offload compiler, CUDA, SuiteSparse with CHOLMOD and METIS, OpenBLAS, and the compiler runtime used by these libraries. The external installation roots are passed explicitly to the build and submission commands.

## Setup

Clone the pinned source:

```bash
bash setup.sh
```

On the target compute-node architecture, load the compiler environment and build:

```bash
SUITESPARSE_ROOT=/path/to/suitesparse \
OPENBLAS_ROOT=/path/to/openblas \
GCC_RUNTIME_ROOT=/path/to/gcc/runtime/lib \
bash build.sh
```

The default build uses `nvc`, targets `cc90`, and selects separate host and device memory. Set `CC` or `GPU_ARCH` when a different compiler command or GPU target is required.

For example, a system using environment modules and dependencies installed below `/opt` can be built with:

```bash
module --force purge
module load nvhpc/25.9-cuda13

SUITESPARSE_ROOT=/opt/suitesparse-metis-7.11.0 \
OPENBLAS_ROOT=/opt/openblas-0.3.32 \
GCC_RUNTIME_ROOT=/opt/gcc-14.3.0/lib64 \
bash build.sh
```

## Run

Submit one small case as a smoke test:

```bash
ACCOUNT=your-slurm-account \
COMPILER_MODULE=your-compiler-module \
SUITESPARSE_ROOT=/path/to/suitesparse \
OPENBLAS_ROOT=/path/to/openblas \
GCC_RUNTIME_ROOT=/path/to/gcc/runtime/lib \
bash slurm/traeff_gpu_memory.sh --task-index 0
```

Submit all 11 meshes by omitting `--task-index 0`. Set `PARTITION` when the GPU partition is not named `gpu`.

Each job samples application RSS and device-memory consumption for 180 seconds. The first 30 seconds are excluded from the reported steady-state statistics. This bounded measurement does not require the optimization to finish.

## Results

Each mesh writes:

```text
benchmarks/results/traeff_2023/openmp/gpu/<mesh>/<run-id>/
├── configuration.toml
└── memory.csv
```

The memory CSV contains the natural host and device peaks as well as post-warmup medians and 95th percentiles. The configuration records the build, numerical parameters, sampling intervals, and bounded measurement window.

## PCG diagnostic

The reported loss of preconditioner positivity can be reproduced independently with the tracked diagnostic patch:

```bash
ACCOUNT=your-slurm-account \
COMPILER_MODULE=your-compiler-module \
SUITESPARSE_ROOT=/path/to/suitesparse \
OPENBLAS_ROOT=/path/to/openblas \
GCC_RUNTIME_ROOT=/path/to/gcc/runtime/lib \
bash slurm/pcg_diagnostic.sh
```

The diagnostic builds a temporary instrumented copy of the pinned source and records every value of $r^{\mathsf T}z$ until the first non-positive value. It does not modify the executable used by the memory campaign.

The diagnostic results are written below `benchmarks/results/traeff_2023/openmp/diagnostics/pcg_positivity/<run-id>/` and include the complete textual trace, `pcg_preconditioner_trace.csv`, `summary.csv`, `configuration.toml`, and the build log.

