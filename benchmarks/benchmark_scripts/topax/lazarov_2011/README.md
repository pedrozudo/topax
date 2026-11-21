# Filters in topology optimization based on Helmholtz-type differential equations

Boyan Stefanov Lazarov and Ole Sigmund

[https://onlinelibrary.wiley.com/doi/10.1002/nme.3072](https://onlinelibrary.wiley.com/doi/10.1002/nme.3072)

```bibtex
@article{lazarov2011filters,
  title={Filters in topology optimization based on Helmholtz-type differential equations},
  author={Lazarov, Boyan Stefanov and Sigmund, Ole},
  journal={International journal for numerical methods in engineering},
  volume={86},
  number={6},
  pages={765--781},
  year={2011},
  publisher={Wiley Online Library}
}
```

## Benchmark

This benchmark reproduces the mesh-independence study from Figure 8 using the Helmholtz PDE density filter with

\[
r = 0.02.
\]

The cantilever has aspect ratio \(4:1\), volume fraction \(0.5\), \(E=1\), \(\nu=0.2\), thickness \(0.01\), SIMP penalty \(p=3\), and a downward point load of magnitude \(0.2\) at the midpoint of the free edge. The left edge is fully fixed. The paper uses an OC update and terminates when the maximum change in the design variables is below \(0.01\).

The three Figure 8 meshes are run sequentially:

```text
paper mesh     topax mesh
100 x 400      400 x 100
200 x 800      800 x 200
300 x 1200     1200 x 300
```

The dimensions are reversed because topax represents the domain as \(4\times1\).

## Running

Run the benchmark script from the repository root:

```bash
python benchmarks/benchmark_scripts/topax/lazarov_2011/cantilever_helmholtz_density.py
```

The script runs all three resolutions and writes densities and summary files to

```text
benchmarks/results/lazarov_2011/figure_8_mesh_independence/
```

Platforms, meshes, and output directories can be selected explicitly:

```bash
python benchmarks/benchmark_scripts/topax/lazarov_2011/cantilever_helmholtz_density.py \
  --device cpu --case 400x100 --output-dir /tmp/lazarov-cpu
```

Repeat `--case` to choose several meshes and use `--device gpu` for a GPU run.
The CSV and JSON summaries contain final convergence and solver data, the
compliance comparison, standardized phase timings, and runtime metadata. A
separate history CSV for each mesh records per-iteration convergence, timing,
and solver data.

For a single selected case, write the iteration history to a specific CSV file
and optionally limit the number of iterations with:

```bash
python benchmarks/benchmark_scripts/topax/lazarov_2011/cantilever_helmholtz_density.py \
  --case 400x100 --runtime-csv runtime.csv --max-iterations 10
```

Three larger exposition cases are also available:

```text
Topax mesh   elements    levels
1600 x 400    640,000      3
3200 x 800  2,560,000      3
6400 x 1600 10,240,000      3
```

The hierarchy depths follow the same progression as the six Sigmund meshes:
the three Figure 8 cases use two levels and the three larger cases use three
levels. The standalone script runs the three Figure 8 paper meshes by default.

After activating a CUDA-enabled Topax environment on the RTX 4090, run all six
meshes with:

```bash
bash benchmarks/benchmark_scripts/topax/scripts/lazarov_gpu.sh
```

The local launcher performs a complete runtime run and a separate 10-iteration
memory run for each mesh. Results are written below
`benchmarks/results/lazarov_2011/topax/gpu/`, while local logs are kept in the
ignored `scripts/.runs/lazarov_2011/` directory.

## Implementation choices

The topax Helmholtz API uses

\[
r_h = 2\sqrt{3}\,r,
\]

so the paper value \(r=0.02\) is passed as

```python
FILTER_RADIUS = 2.0 * np.sqrt(3.0) * 0.02
```

The optimization uses

```python
topax.normalized_volume_constraint
```

which evaluates

\[
\frac{\operatorname{mean}(\rho)}{\mathrm{volfrac}} - 1.
\]

The compliance objective is normalized by the compliance of the initial uniform design before being passed to OC. This only changes the numerical scale of the objective and gradient; it does not change the OC solution.

A design-variable lower bound of

```python
DESIGN_RHO_MIN = 1e-3
```

is used. The corresponding topax modified-SIMP stiffness floor is

```python
RHO_MIN = 1e-9
```

for \(p=3\). Keeping the design variables strictly positive also avoids permanently locking variables at zero in the multiplicative OC update.

The paper reports a thickness of \(0.01\), while the topax 2D elasticity formulation includes thickness explicitly in the stiffness matrix. The benchmark therefore keeps the raw topax compliance separate from the thickness-scaled compliance used for comparison with Figure 8.

The paper notes that these nominally two-dimensional examples are modeled with three-dimensional hexahedral elements, whereas this benchmark uses the Topax 2D elasticity formulation.

The OC move limit is set to

```python
OC_MOVE = 0.2
```

as a reproduction choice because the paper specifies OC but does not provide all numerical details of the update. The stopping criterion follows the paper:

```python
max(abs(rho_new - rho)) < 0.01
```

All cases may also stop after at least 500 iterations when the relative
compliance change over 100 iterations is at most `1e-5`. The result records
`stopping_reason` as `design_change`, `compliance_plateau`, or
`iteration_limit`.

The six cases use `L2, L2, L2, L3, L3, L3` in increasing mesh-size order.
The corresponding crossover configurations are `L2-C1` for the first three
meshes and `L3-C2` for the remaining meshes.
