# Träff et al. 2023 Benchmark

This example reproduces the large-scale 3D cantilever topology-optimization benchmark from:

> E. A. Träff, A. Rydahl, S. Karlsson, O. Sigmund, N. Aage,  
> *Simple and efficient GPU accelerated topology optimisation: Codes and applications*, 
> Computer Methods in Applied Mechanics and Engineering 410 (2023) 116043  
> DOI: `10.1016/j.cma.2023.116043`



## Problem

The benchmark uses:

- `320 × 160 × 160` hexahedral elements
- 8,192,000 elements
- approximately 25 million displacement DOFs
- volume fraction `0.1`
- SIMP penalty `p = 3`
- `E0 = 1`
- `Emin = 1e-6`
- Poisson ratio `0.3`
- density-filter radius `1.5` elements
- optimality-criteria design updates
- PCG tolerance `1e-5`
- four pre- and four post-Jacobi sweeps with `ω = 0.6`
- 20 design iterations

The load and support geometry follow the released Träff OpenMP-GPU implementation. A distributed circular traction patch acts in the negative z direction, while two strips on the opposite face are clamped.


The current default is the 8.2-million-element grid used for the smoothing
ablation. The largest published timing point uses `640 × 320 × 320` elements.

## Running

Run the four-level benchmark with:

```bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py --levels 4
```

Run the five-level configuration with:

```bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py --levels 5
```

Run the largest published grid with:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py \
    --elements 640 320 320 --levels 5
```

To also capture a JAX/TensorBoard profile:

```bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py --levels 5 --profile
```



Run the 1+1, 2+2, and 4+4 smoothing comparison with:

```bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py --levels 4 --smoothing-steps 1
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py --levels 4 --smoothing-steps 2
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py --levels 4 --smoothing-steps 4
```

Each smoothing count is compiled and warmed up independently. Compare only
the reported design-iteration time and CG counts.

The Galerkin representation crossover is independently configurable. Level 0
is the fine grid. For a four-level hierarchy, materialization only on the
coarsest grid is selected by:

```bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py --levels 4 --galerkin-crossover-level 3
```

To materialize level 2 and recursively construct level 3 from it, use:

```bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py --levels 4 --galerkin-crossover-level 2
```

Every implicit Galerkin level uses convolutional Q1 patch extraction and
structured einops assembly, without element-to-DOF connectivity indices or a
selectable backend.

The implicit Galerkin action is evaluated in bounded spatial slabs so that
element-local values and contributions are not constructed for the complete
grid at once.

Report compiler memory for the canonical benchmark with:

```bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py \
    --levels 4 --report-compiler-memory
```

### Fine-grid matrix-free implementation

The benchmark uses Topax's canonical slabbed matrix-free operator. It bounds
the convolution intermediate by evaluating rematerialized nodal slabs and
selects the largest element dimension as the slab axis.

The fine-grid slab configuration is exposed independently of the multigrid
configuration:

~~~bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py \
    --elements 640 320 320 \
    --levels 5 \
    --slab-size 24 \
    --slab-axis 0 \
    --report-compiler-memory
~~~

The slab-axis option uses 0=x, 1=y, and 2=z; omitting it selects the largest
element dimension. These options are applied consistently to the fine-grid
CG operator, the level-zero V-cycle smoother, and the compliance evaluation.
The benchmark header reports the resolved axis and slab count. Compiler
memory output also includes the total estimate: temporary + arguments +
output - aliased.

## Timing

The example separately reports:

- setup time
- JAX compilation time
- dummy/warmup execution time
- total design-iteration time
- average design-iteration time
- iterations per second
- total PCG iteration count
- average PCG iteration count
- PCG relative residual

For comparison with the published Träff timings, use the reported `design iterations` and `average iteration` values.

Do not include setup, JAX compilation, or warmup time in the steady-state comparison.

Per-design-iteration measurements can be written directly to CSV with:

```bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py \
    --runtime-csv runtime.csv
```



## Profiling

A JAX profile can be recorded with:

```bash
python benchmarks/benchmark_scripts/topax/traeff_2023/cantilever.py --levels 5 --profile
```

The first few steady-state design iterations are written to a TensorBoard-compatible profile directory under `/tmp`.

