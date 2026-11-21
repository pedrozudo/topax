# Wang et al. 2025 benchmark

This benchmark reproduces the cantilever topology-optimization problem from
Fig. 10 and Table 1 of:

> J. Wang, N. Aage, J. Wu, O. Sigmund, R. Westermann,
> *Efficient large-scale 3D topology optimization with matrix-free MATLAB
> code*, Structural and Multidisciplinary Optimization 68 (2025), article 174.
> DOI: [10.1007/s00158-025-04127-3](https://doi.org/10.1007/s00158-025-04127-3)

The authors' reference implementation is
[TOP3D_XL](https://github.com/PSLer/TOP3D_XL).

## Problem

The fixed `1.0 × 0.5 × 0.5` design domain uses:

- volume fraction `0.12`;
- `E0 = 1`, `Emin = 1e-6`, Poisson ratio `0.3`;
- SIMP penalty `3`;
- 50 optimality-criteria iterations with move limit `0.2`;
- MGPCG tolerance `1e-3` and maximum 600 iterations;
- a PDE sensitivity filter;
- all displacement components fixed on `x = 0`;
- total force `Fz = -1`, distributed uniformly over the bottom sixth of the
  `x = 1` face.

The density enters the elasticity problem directly. Following TOP3D_XL, only
the compliance sensitivity is filtered:

```text
filtered_dc = PDE(rho * dc) / max(1e-3, rho)
```

This is intentionally implemented inside the benchmark rather than through
Topax's high-level Helmholtz density-filter formulation.

## Resolutions

The benchmark contains the published resolutions and an additional R1024
scaling case:

| Case | Elements | Count |
|---|---:|---:|
| R48 | `48 × 24 × 24` | 27,648 |
| R96 | `96 × 48 × 48` | 221,184 |
| R192 | `192 × 96 × 96` | 1,769,472 |
| R300 | `300 × 150 × 150` | 6,750,000 |
| R384 | `384 × 192 × 192` | 14,155,776 |
| R768 | `768 × 384 × 384` | 113,246,208 |
| R800 | `800 × 400 × 400` | 128,000,000 |
| R1024 | `1024 × 512 × 512` | 268,435,456 |

R300 is listed by the CLI but deliberately rejected. Its transverse element
counts admit only one exact factor-of-two coarsening, leaving a coarse problem
far too large for Topax's direct coarse solver. The other cases use exact
dyadic hierarchies. R1024 extends the published series with approximately
twice as many elements as R800.

The GPU submission script disables JAX memory preallocation for R1024 so that
the complete device memory remains available to JAX and the external cuDSS
factorization. Smaller runtime cases use a preallocated fraction of `0.75`.

## Running

Start with the default R48 case:

```bash
python benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py
```

Select a paper resolution and optionally shorten a validation run:

```bash
python benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py \
    --resolution 192 --max-iterations 5
```

Run the 128-million-element target:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
python benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py \
    --resolution 800 --levels 5
```

The implicit Galerkin matvec and Jacobi diagonal are evaluated in spatial
slabs. The default width is four coarse elements and can be adjusted without
changing the hierarchy or solver tolerances:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
python benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py \
    --resolution 800 \
    --levels 5 \
    --galerkin-crossover-level 4 \
    --galerkin-slab-size 4 \
    --max-iterations 5 \
    --report-compiler-memory
```

Compliance is evaluated as the force--displacement product. Its analytical
SIMP sensitivity is computed from element strain energies in matrix-free
slabs controlled by `--slab-size`.
The fine-grid stiffness action uses the same slab width, extracts boundary
halos without padding the complete fields, and writes each slab directly into
its owned nodal planes.

This is a scaling target, not an expectation that the complete optimization
fits on every GPU. R800 has 386,404,803 elastic DOFs; one FP64 vector of that
shape is approximately 2.88 GiB. The persistent PCG working set contains
several such vectors before JAX temporaries, density/filter state, and the
multigrid hierarchy are counted.

Report compiler-estimated memory or capture a profile with:

```bash
python benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py \
    --resolution 48 --report-compiler-memory

python benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py \
    --resolution 48 --profile
```

Per-design-iteration measurements can be written directly to CSV with:

```bash
python benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py \
    --resolution 48 --runtime-csv runtime.csv
```

## Mesh-independent filter series

The normal experiment uses `r = sqrt(3)` elements at every resolution. Pass
`--fixed-filter` for the physically fixed series from Fig. 10b/Table 1:

```bash
python benchmarks/benchmark_scripts/topax/wang_2025/cantilever.py \
    --resolution 384 --fixed-filter
```

The filter-radius multipliers are `1, 2, 4, 6, 8, 16, 17` for
R48, R96, R192, R300, R384, R768, and R800 respectively.

## Solver distinction

The physical and optimization problem follows Wang et al., but the linear
systems use Topax's exact structured Galerkin MGPCG implementation. The paper
uses an adapted non-dyadic V-cycle with diagonal relaxation. Consequently,
Table 1 compliance and MDN values are printed as literature targets, while
solver timing and iteration counts should be interpreted as Topax results.
