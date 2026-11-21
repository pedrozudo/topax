# Benchmark evaluation

## 2D

Evaluate the Lazarov and Sigmund GPU campaigns independently:

```bash
python benchmarks/evaluation/2D/lazarov_2011/evaluate.py
python benchmarks/evaluation/2D/lazarov_2011/plot_designs.py
python benchmarks/evaluation/2D/sigmund_2007/evaluate.py
python benchmarks/evaluation/2D/sigmund_2007/plot_designs.py
```

Each evaluator validates one complete six-mesh campaign and writes a CSV and LaTeX table in its benchmark-specific directory. Each plotting command creates a two-by-three PDF containing the corresponding final physical densities.

## 3D

The Träff GPU evaluator reads the timestamped raw results, validates mesh coverage and measurement scope, and writes the quantities used by the plots:

```bash
python benchmarks/evaluation/3D/traeff_2023/evaluate_gpu.py
```

This produces `benchmarks/evaluation/3D/traeff_2023/traeff_gpu.csv`. If more than one complete campaign exists for a configuration, select its run ID with the corresponding option shown by `--help`.

After completing the dedicated Topax steady-memory sweep, create the combined host/device comparison with the OpenMP implementation using:

```bash
python benchmarks/evaluation/3D/traeff_2023/evaluate_steady_memory.py
```

This validates both complete 11-mesh campaigns and writes `benchmarks/evaluation/3D/traeff_2023/traeff_memory_comparison.csv`. Use `--topax-run` or `--openmp-run` if more than one complete campaign is present.

Plotting requires Matplotlib. Generate both sets of figures with:

```bash
python benchmarks/evaluation/3D/traeff_2023/plot_gpu.py
```

The implementation comparison uses Topax L4-C2, Futtop, and pyFANTOM and produces `traeff_gpu_runtime.pdf`, `traeff_gpu_memory.pdf`, and `traeff_gpu_pcg.pdf`. The Topax configuration comparison uses L4-C2 and L4-C3 and produces the corresponding `traeff_gpu_topax_*.pdf` files. All generated files are kept in `benchmarks/evaluation/3D/traeff_2023/`.

Evaluate the Topax Wang GPU runs with:

```bash
python benchmarks/evaluation/3D/wang_2025/evaluate_gpu.py
```

Generate the L4-C3 and L5-C4 comparison with:

```bash
python benchmarks/evaluation/3D/wang_2025/plot_gpu.py
```

The evaluator treats R1024 as an L5-C4-only mesh. The CSV and the three runtime, memory, and PCG plots are written to `benchmarks/evaluation/3D/wang_2025/`.

Evaluate the currently available Topax and TOP3D_XL Wang CPU runs with:

```bash
python benchmarks/evaluation/3D/wang_2025/evaluate_cpu.py
```

Generate the partial CPU comparison with:

```bash
python benchmarks/evaluation/3D/wang_2025/plot_cpu.py
```

The CPU evaluator accepts incomplete campaigns and includes newly completed meshes whenever it is rerun. R768 is retained in the CSV but omitted from the figures. At each shared plotted resolution, the runtime figure annotates the L5-C4 and L4-C3 speedups over TOP3D_XL in their corresponding line colors.
