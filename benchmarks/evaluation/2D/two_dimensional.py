"""Shared evaluation logic for the two-dimensional Topax benchmarks."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


BENCHMARKS_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = BENCHMARKS_DIR / "results"
DEFAULT_EVALUATION_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    short_name: str
    meshes: tuple[tuple[int, int], ...]
    levels: tuple[int, ...]
    crossover_levels: tuple[int, ...]
    table_caption: str
    table_label: str

    @property
    def output_dir(self) -> Path:
        return DEFAULT_EVALUATION_ROOT / self.key


SPECS = {
    "lazarov_2011": BenchmarkSpec(
        key="lazarov_2011",
        short_name="lazarov",
        meshes=(
            (400, 100),
            (800, 200),
            (1200, 300),
            (1600, 400),
            (3200, 800),
            (6400, 1600),
        ),
        levels=(2, 2, 2, 3, 3, 3),
        crossover_levels=(1, 1, 1, 2, 2, 2),
        table_caption=(
            "Optimization and computational results for the mesh-refined "
            "cantilever problems."
        ),
        table_label="tab:lazarov-2d",
    ),
    "sigmund_2007": BenchmarkSpec(
        key="sigmund_2007",
        short_name="sigmund",
        meshes=(
            (120, 40),
            (360, 120),
            (750, 250),
            (1500, 500),
            (3000, 1000),
            (5400, 1800),
        ),
        levels=(2, 2, 2, 3, 3, 3),
        crossover_levels=(1, 1, 1, 2, 2, 2),
        table_caption=(
            "Optimization and computational results for the geometrically "
            "scaled MBB-beam problems."
        ),
        table_label="tab:sigmund-2d",
    ),
}

OUTPUT_FIELDS = (
    "benchmark",
    "run_id",
    "mesh",
    "nx",
    "ny",
    "n_elements",
    "levels",
    "galerkin_crossover_level",
    "design_iterations",
    "average_iteration_time_s",
    "mean_physics_pcg_iterations",
    "memory_design_iterations",
    "gpu_peak_above_baseline_gib",
    "final_compliance",
    "non_discreteness_percent",
    "stopping_reason",
)


def mesh_name(mesh: tuple[int, int]) -> str:
    return "x".join(map(str, mesh))


def read_configuration(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not reader.fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    return rows


def number(row: dict[str, str], field: str, path: Path) -> float:
    value = row.get(field, "")
    if value is None or not value.strip():
        raise ValueError(f"empty {field!r} in {path}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field!r} in {path}")
    return result


def matches_configuration(
    configuration: dict,
    spec: BenchmarkSpec,
    mesh: tuple[int, int],
    levels: int,
    crossover: int,
) -> bool:
    runtime = configuration.get("measurements", {}).get("runtime", {})
    memory = configuration.get("measurements", {}).get("memory", {})
    return (
        configuration.get("schema_version") == 2
        and configuration.get("benchmark") == spec.key
        and configuration.get("implementation") == "topax"
        and configuration.get("backend") == "gpu"
        and tuple(configuration.get("elements", ())) == mesh
        and configuration.get("levels") == levels
        and configuration.get("galerkin_crossover_level") == crossover
        and configuration.get("precision") == "float64"
        and runtime.get("design_iteration_limit") == 10000
        and runtime.get("full_benchmark") is True
        and memory.get("design_iteration_limit") == 10
        and memory.get("full_benchmark") is False
        and memory.get("jax_preallocate") is False
    )


def select_campaign(
    spec: BenchmarkSpec,
    results_root: Path,
    requested_run_id: str | None,
) -> tuple[str, dict[tuple[int, int], Path]]:
    root = results_root / spec.key / "topax" / "gpu"
    if not root.is_dir():
        raise ValueError(f"missing result directory: {root}")

    matches = {}
    for mesh, levels, crossover in zip(
        spec.meshes,
        spec.levels,
        spec.crossover_levels,
    ):
        mesh_root = root / mesh_name(mesh)
        if not mesh_root.is_dir():
            raise ValueError(f"missing mesh directory: {mesh_root}")
        mesh_matches = {}
        for run_dir in sorted(mesh_root.iterdir()):
            configuration_path = run_dir / "configuration.toml"
            required = (
                run_dir / "runtime.csv",
                run_dir / "memory.csv",
                run_dir / "artifacts" / "summary.csv",
            )
            if (
                run_dir.is_dir()
                and configuration_path.is_file()
                and all(path.is_file() for path in required)
                and matches_configuration(
                    read_configuration(configuration_path),
                    spec,
                    mesh,
                    levels,
                    crossover,
                )
            ):
                mesh_matches[run_dir.name] = run_dir
        if not mesh_matches:
            raise ValueError(
                f"no matching {spec.key} result found for {mesh_name(mesh)}"
            )
        matches[mesh] = mesh_matches

    common_ids = set.intersection(*(set(runs) for runs in matches.values()))
    if requested_run_id is not None:
        if requested_run_id not in common_ids:
            raise ValueError(
                f"run {requested_run_id} does not contain all six "
                f"{spec.key} meshes"
            )
        run_id = requested_run_id
    elif len(common_ids) == 1:
        run_id = common_ids.pop()
    elif not common_ids:
        raise ValueError(
            f"no single {spec.key} run ID contains all six meshes"
        )
    else:
        choices = ", ".join(sorted(common_ids))
        raise ValueError(
            f"multiple complete {spec.key} runs found ({choices}); "
            "select one with --run-id"
        )
    return run_id, {mesh: runs[run_id] for mesh, runs in matches.items()}


def evaluate_case(
    spec: BenchmarkSpec,
    run_id: str,
    mesh: tuple[int, int],
    levels: int,
    crossover: int,
    run_dir: Path,
) -> dict[str, str | int | float]:
    runtime_path = run_dir / "runtime.csv"
    runtime_rows = read_csv(runtime_path)
    iterations = [
        int(number(row, "iteration", runtime_path)) for row in runtime_rows
    ]
    if iterations != list(range(1, len(runtime_rows) + 1)):
        raise ValueError(f"nonconsecutive iterations in {runtime_path}")
    iteration_times = [
        number(row, "iteration_time_s", runtime_path) for row in runtime_rows
    ]
    pcg_iterations = [
        number(row, "physics_cg_iterations", runtime_path)
        for row in runtime_rows
    ]
    if not runtime_rows or min(iteration_times) < 0.0 or min(pcg_iterations) < 0.0:
        raise ValueError(f"invalid runtime history in {runtime_path}")

    memory_path = run_dir / "memory.csv"
    memory_rows = read_csv(memory_path)
    if len(memory_rows) != 1:
        raise ValueError(f"expected one memory row in {memory_path}")
    memory_row = memory_rows[0]
    memory_iterations = int(
        number(memory_row, "design_iterations_completed", memory_path)
    )
    if memory_iterations != 10:
        raise ValueError(f"memory run did not complete 10 iterations: {memory_path}")
    peak_memory = number(
        memory_row, "gpu_peak_above_baseline_gib", memory_path
    )
    if peak_memory < 0.0:
        raise ValueError(f"negative GPU memory value in {memory_path}")

    summary_path = run_dir / "artifacts" / "summary.csv"
    summary_rows = read_csv(summary_path)
    if len(summary_rows) != 1:
        raise ValueError(f"expected one summary row in {summary_path}")
    summary = summary_rows[0]
    if summary.get("benchmark") != spec.key:
        raise ValueError(f"unexpected benchmark in {summary_path}")
    if summary.get("case") != mesh_name(mesh):
        raise ValueError(f"unexpected mesh in {summary_path}")
    if int(number(summary, "iterations", summary_path)) != len(runtime_rows):
        raise ValueError(f"iteration count mismatch in {summary_path}")
    converged = summary.get("converged", "").lower() == "true"
    stopping_reason = summary.get("stopping_reason", "")
    if not converged and not (
        stopping_reason == "iteration_limit" and len(runtime_rows) == 10000
    ):
        raise ValueError(f"invalid stopping state in {summary_path}")

    average_time = statistics.fmean(iteration_times)
    average_pcg = statistics.fmean(pcg_iterations)
    if not math.isclose(
        average_time,
        number(summary, "average_iteration_time_s", summary_path),
        rel_tol=1e-12,
    ):
        raise ValueError(f"average runtime mismatch in {summary_path}")
    if not math.isclose(
        average_pcg,
        number(summary, "average_physics_cg_iterations", summary_path),
        rel_tol=1e-12,
    ):
        raise ValueError(f"average PCG count mismatch in {summary_path}")

    nx, ny = mesh
    return {
        "benchmark": spec.key,
        "run_id": run_id,
        "mesh": mesh_name(mesh),
        "nx": nx,
        "ny": ny,
        "n_elements": nx * ny,
        "levels": levels,
        "galerkin_crossover_level": crossover,
        "design_iterations": len(runtime_rows),
        "average_iteration_time_s": average_time,
        "mean_physics_pcg_iterations": average_pcg,
        "memory_design_iterations": memory_iterations,
        "gpu_peak_above_baseline_gib": peak_memory,
        "final_compliance": number(summary, "final_compliance", summary_path),
        "non_discreteness_percent": number(
            summary, "mnd_percent", summary_path
        ),
        "stopping_reason": stopping_reason,
    }


def evaluate_campaign(
    spec: BenchmarkSpec,
    results_root: Path,
    requested_run_id: str | None,
) -> tuple[str, list[dict[str, str | int | float]]]:
    run_id, run_dirs = select_campaign(spec, results_root, requested_run_id)
    rows = [
        evaluate_case(
            spec,
            run_id,
            mesh,
            levels,
            crossover,
            run_dirs[mesh],
        )
        for mesh, levels, crossover in zip(
            spec.meshes,
            spec.levels,
            spec.crossover_levels,
        )
    ]
    return run_id, rows


def latex_table(spec: BenchmarkSpec, rows: list[dict]) -> str:
    body = []
    for row in rows:
        mesh = str(row["mesh"]).replace("x", "$\\times$")
        values = (
            mesh,
            f"{int(row['n_elements']):,}",
            str(int(row["levels"])),
            f"{int(row['design_iterations']):,}",
            f"{float(row['average_iteration_time_s']):.3f}",
            f"{float(row['mean_physics_pcg_iterations']):.2f}",
            f"{float(row['gpu_peak_above_baseline_gib']):.3f}",
            f"{float(row['final_compliance']):.4f}",
            f"{float(row['non_discreteness_percent']):.3f}",
        )
        body.append(" & ".join(values) + r" \\")

    return rf"""\begin{{table*}}[t]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\caption{{{spec.table_caption}}}
\label{{{spec.table_label}}}
\begin{{tabular}}{{rrrrrrrrr}}
\toprule
mesh &
elements &
levels &
design iter. &
\shortstack{{avg. time per iter. [s]}} &
\shortstack{{avg. PCG iter.}} &
\shortstack{{peak mem. [GiB]}} &
compliance &
\shortstack{{non-discreteness \mbox{{[\%]}}}} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table*}}"""


def evaluation_main(benchmark: str) -> None:
    spec = SPECS[benchmark]
    parser = argparse.ArgumentParser(
        description=f"Evaluate the Topax {benchmark} GPU campaign."
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=spec.output_dir)
    parser.add_argument("--run-id", help="explicit complete campaign run ID")
    args = parser.parse_args()

    try:
        run_id, rows = evaluate_campaign(
            spec,
            args.results_root.expanduser().resolve(),
            args.run_id,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{spec.short_name}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    table_path = output_dir / f"{spec.short_name}_table.tex"
    table_path.write_text(latex_table(spec, rows) + "\n", encoding="utf-8")
    print(f"topax: {run_id}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {table_path}")


def design_main(benchmark: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    spec = SPECS[benchmark]
    parser = argparse.ArgumentParser(
        description=f"Plot the final Topax {benchmark} designs."
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=spec.output_dir)
    parser.add_argument("--run-id", help="explicit complete campaign run ID")
    args = parser.parse_args()

    try:
        run_id, run_dirs = select_campaign(
            spec,
            args.results_root.expanduser().resolve(),
            args.run_id,
        )
        densities = []
        for mesh in spec.meshes:
            summary_path = run_dirs[mesh] / "artifacts" / "summary.csv"
            summary_rows = read_csv(summary_path)
            if len(summary_rows) != 1:
                raise ValueError(f"expected one summary row in {summary_path}")
            filename = summary_rows[0].get("physical_density_file", "")
            density_path = run_dirs[mesh] / "artifacts" / filename
            if not filename or not density_path.is_file():
                raise ValueError(f"missing physical density: {density_path}")
            density = np.load(density_path, mmap_mode="r")
            if density.shape != mesh:
                raise ValueError(
                    f"expected density shape {mesh}, found {density.shape}: "
                    f"{density_path}"
                )
            densities.append(density)
    except ValueError as error:
        raise SystemExit(str(error)) from None

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 3.0))
    for axis, mesh, density in zip(axes.flat, spec.meshes, densities):
        axis.imshow(
            density.T,
            origin="lower",
            cmap="gray_r",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        axis.set_axis_off()
        axis.text(
            0.98,
            0.95,
            f"{mesh[0]} × {mesh[1]}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
                "pad": 1.5,
            },
        )

    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.01,
        top=0.99,
        wspace=0.02,
        hspace=0.08,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{spec.short_name}_final_designs.pdf"
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"topax: {run_id}")
    print(f"Wrote {output}")
