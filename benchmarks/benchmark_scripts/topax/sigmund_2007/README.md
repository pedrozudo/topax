# Morphology-based black and white filters for topology optimization

Ole Sigmund

https://link.springer.com/article/10.1007/s00158-006-0087-x


```bibtex
@article{sigmund2007morphology,
  title={Morphology-based black and white filters for topology optimization},
  author={Sigmund, Ole},
  journal={Structural and Multidisciplinary Optimization},
  volume={33},
  number={4},
  pages={401--424},
  year={2007},
  publisher={Springer}
}
```

## Running

Run the example from the repository root:

```bash
python benchmarks/benchmark_scripts/topax/sigmund_2007/mbb_conv_filter.py
```

The script writes the design and physical-density arrays to
`benchmarks/results/sigmund_2007/`.

Select a platform and output directory explicitly with

```bash
python benchmarks/benchmark_scripts/topax/sigmund_2007/mbb_conv_filter.py \
  --device cpu --case 120x40 --output-dir /tmp/sigmund-cpu
```

Use `--device gpu` for a GPU run. In addition to the density arrays, the script
writes `summary.csv`, `summary.json`, and `history.csv`. They contain the
compliance and discreteness validation, per-iteration convergence and solver
data, standardized phase timings, and runtime metadata.

Write the iteration history to a specific CSV file and optionally limit the
number of iterations with:

```bash
python benchmarks/benchmark_scripts/topax/sigmund_2007/mbb_conv_filter.py \
  --case 120x40 --runtime-csv runtime.csv --max-iterations 10
```

The original `120x40` literature case is the default and uses two multigrid
levels. The scaled exposition cases retain the 3:1 aspect ratio, unit-sized
elements, and the 3.5-element filter radius:

```text
360x120      43,200 elements       L2
750x250     187,500 elements       L2
1500x500    750,000 elements       L3
3000x1000 3,000,000 elements       L3
5400x1800 9,720,000 elements       L3
```

The crossover is the final level in each hierarchy, giving `L2-C1` for the
first three meshes and `L3-C2` for the remaining meshes.

Only the `120x40` case has published compliance and non-discreteness reference
values; the larger cases are performance and final-design experiments.

The optimization stops when the maximum design change is below `0.01` or when
the relative compliance change is at most `1e-5` over 100 iterations after at
least 500 iterations. The maximum is 10,000 iterations. The original
`120x40` case satisfies the design-change criterion; the plateau criterion
provides a non-arbitrary endpoint for scaled cases whose density updates keep
oscillating after their compliance has stabilized. The result records the
stopping reason and both the actual and literature iteration counts.

After activating a CUDA-enabled Topax environment on the RTX 4090, run all six
meshes with:

```bash
bash benchmarks/benchmark_scripts/topax/scripts/sigmund_gpu.sh
```

The local launcher performs a complete runtime run and a separate 10-iteration
memory run for each mesh. Results are written below
`benchmarks/results/sigmund_2007/topax/gpu/`, while local logs are kept in the
ignored `scripts/.runs/sigmund_2007/` directory.
