# Futtop benchmarks

This directory contains the setup and submission scripts used to run the Träff et al. three-dimensional cantilever benchmark with [Futtop](https://github.com/topopt/futtop) v0.9.1. The external source is checked out at commit `c3cd2b00632ad57cd3d8917246552b2fb4e99e01` and is not tracked by Git.

## Prerequisites

The setup requires Git and Futhark 0.21.11. Building the generated code requires a C99 compiler with OpenMP support and OpenCL development headers and libraries. Futhark is treated as an external prerequisite and its installation root is passed to the setup command.

## Setup

On a machine capable of running the Futhark compiler, run:

```bash
FUTHARK_HOME=/path/to/futhark-0.21.11 bash setup.sh
```

The script clones Futtop into `Futtop/`, applies the tracked patch that disables VTR output during benchmarking, and generates the OpenCL C library. It does not install or modify the Futhark compiler.

Compile the generated C code on the target compute-node architecture:

```bash
CPPFLAGS="-I/path/to/opencl/include" \
OPENCL_LIBS="-L/path/to/opencl/lib -lOpenCL" \
bash build.sh
```

The two variables may be omitted when the OpenCL headers and library are available through the standard compiler search paths. `CC` and `CFLAGS` may also be set to select a compiler or additional optimization flags.

Some systems provide only the versioned OpenCL loader `libOpenCL.so.1`, without an unversioned `libOpenCL.so` linker file. In that case, use:

```bash
CPPFLAGS="-I/path/to/opencl/include" \
OPENCL_LIBS="-L/path/to/opencl/lib -l:libOpenCL.so.1" \
bash build.sh
```

## Run

Submit a single small case as a smoke test:

```bash
ACCOUNT=your-slurm-account bash slurm/traeff_gpu.sh --task-index 0
```

Submit all 11 meshes:

```bash
ACCOUNT=your-slurm-account bash slurm/traeff_gpu.sh
```

Set `PARTITION` as well when the Slurm installation does not use `gpu` as the GPU partition name. By default, the launcher submits separate runtime and memory arrays under one run ID.

## Results

Each mesh writes:

```text
benchmarks/results/traeff_2023/futtop/gpu/<mesh>/<run-id>/
├── configuration.toml
├── runtime.csv
└── memory.csv
```

The configuration records the Futtop and Futhark revisions, solver parameters, precision, iteration scope, and memory-measurement settings associated with each CSV file.

