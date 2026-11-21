# pyFANTOM benchmarks

This directory contains the setup and submission scripts used to run the Träff et al. three-dimensional cantilever benchmark with [pyFANTOM](https://github.com/ahnobari/pyFANTOM) 0.1.0. The setup checks out pyFANTOM at commit `ffa4c95f087f86933629cb1926030b80501c3136`; the external source and Python environment are installed locally and are not tracked by Git.

## Prerequisites

The setup requires Python 3.9, Git, CUDA 13, and a C/C++ compiler. A SuiteSparse installation providing CHOLMOD and METIS must already be available, including its headers and shared libraries. The CUDA and SuiteSparse installation roots are passed explicitly in the commands below.

## Setup

From this directory, run:

```bash
PYTHON=python3.9 \
CUDA_HOME=/path/to/cuda \
CHOLMOD_ROOT=/path/to/suitesparse \
bash setup.sh
```

`CHOLMOD_ROOT` may be omitted when SuiteSparse is available through the standard compiler and dynamic-linker search paths. The script creates `.venv/`, clones pyFANTOM into `pyFANTOM/`, installs CuPy 13.6.0 for CUDA 13, builds scikit-sparse against CHOLMOD, and applies the tracked instrumentation needed to record PCG convergence without changing the numerical algorithm.

## Run

Submit a single small case as a smoke test:

```bash
ACCOUNT=your-slurm-account \
CUDA_HOME=/path/to/cuda \
CHOLMOD_ROOT=/path/to/suitesparse \
bash slurm/traeff_gpu.sh --task-index 0
```

Submit all 11 meshes:

```bash
ACCOUNT=your-slurm-account \
CUDA_HOME=/path/to/cuda \
CHOLMOD_ROOT=/path/to/suitesparse \
bash slurm/traeff_gpu.sh
```

Set `PARTITION` as well when the Slurm installation does not use `gpu` as the GPU partition name. By default, the launcher submits separate runtime and memory arrays under one run ID.

## Results

Each mesh writes:

```text
benchmarks/results/traeff_2023/pyfantom/gpu/<mesh>/<run-id>/
├── configuration.toml
├── runtime.csv
└── memory.csv
```

The configuration records the pyFANTOM revision, solver parameters, precision, iteration scope, and allocator settings associated with each CSV file.
