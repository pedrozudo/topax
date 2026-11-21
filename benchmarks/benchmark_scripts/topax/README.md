# Benchmark scripts

The problem definitions are organized by publication. All launchers are kept
in `scripts/`, with scheduled launchers nested below `scripts/slurm/`. Each
invocation receives an immutable UTC run ID of the form `YYYYMMDDTHHMMSSZ`.

```text
lazarov_2011/
sigmund_2007/
traeff_2023/
wang_2025/
scripts/
scripts/slurm/
```

Results are organized by benchmark, implementation, backend, problem instance,
and run ID:

```text
benchmarks/results/<benchmark>/topax/<backend>/<mesh>/<run-id>/
├── configuration.toml
├── runtime.csv
└── memory.csv
```

Three-dimensional GPU runtime and memory measurements use separate Slurm arrays
because memory measurement disables JAX device-memory preallocation. The local
two-dimensional launchers use separate sequential processes for the same
reason. Both measurements use the same run ID and write complementary files.
The Wang CPU script obtains runtime and peak resident memory from one execution.

The `[measurements.runtime]` and `[measurements.memory]` tables in
`configuration.toml` record the iteration limit, whether the measurement spans
the complete benchmark, and the JAX allocator settings used by GPU runs.

The scripts refuse to overwrite an existing measurement. Scheduler output for
the three-dimensional experiments is written to the ignored working directory
`benchmark_scripts/topax/scripts/.slurm/`. Local two-dimensional logs are
written below `benchmark_scripts/topax/scripts/.runs/`, separated by
benchmark. Neither is included in the result data.

The available benchmark launchers are:

```text
scripts/lazarov_gpu.sh
scripts/sigmund_gpu.sh
scripts/slurm/traeff_gpu.sh
scripts/slurm/traeff_gpu_steady_memory.sh
scripts/slurm/wang_gpu.sh
scripts/slurm/wang_cpu.sh
scripts/slurm/run_3d.sh
```

For example:

```bash
benchmarks/benchmark_scripts/topax/scripts/slurm/traeff_gpu.sh \
    --levels 4 --crossover-level 2
```

By default, a three-dimensional GPU invocation submits both runtime and memory
arrays. The local two-dimensional launchers likewise perform both measurements
sequentially. Use `--measurement runtime` or `--measurement memory` to perform
only one. Pass `--run-id` when two separate invocations should contribute to
the same result directory.

The combined host/device steady-state comparison with the released OpenMP
implementation uses a separate phase-resolved sweep. It runs the natural and
initialization-cleanup variants sequentially for every mesh and writes both
rows to `steady_memory.csv`:

```bash
benchmarks/benchmark_scripts/topax/scripts/slurm/traeff_gpu_steady_memory.sh
```

After activating a CUDA-enabled Topax environment on the local GPU machine,
run the two-dimensional experiments independently with:

```bash
bash benchmarks/benchmark_scripts/topax/scripts/lazarov_gpu.sh
bash benchmarks/benchmark_scripts/topax/scripts/sigmund_gpu.sh
```

Each local command runs its six meshes sequentially and creates an independent
run ID and result tree. For every mesh it performs a complete runtime run and a
separate 10-iteration memory run. Set `PYTHON` to select a Python executable.
Use `--measurement runtime` or `--measurement memory` to run one measurement;
pass the original `--run-id` when completing an existing result tree.

Submit the complete three-dimensional experiment set separately with:

```bash
benchmarks/benchmark_scripts/topax/scripts/slurm/run_3d.sh
```

`run_3d.sh` submits the Träff GPU L4-C2 and L4-C3 sweeps and the Wang CPU/GPU
L4-C3 and L5-C4 sweeps. The Wang GPU L4-C3 sweep ends at R800 because its
R1024 coarse factorization does not fit in device memory; the L5-C4 sweep
includes R1024. The underlying Slurm arrays are independent and may run
concurrently.
