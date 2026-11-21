#!/usr/bin/env python3
"""Reduce the Topax Wang GPU runs to paper-facing quantities."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - needed only on Python < 3.11
    import tomli as tomllib


BENCHMARKS_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = (
    BENCHMARKS_DIR / "results" / "wang_2025" / "topax" / "gpu"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "wang_gpu.csv"

COMMON_MESHES = (
    (96, 48, 48),
    (192, 96, 96),
    (384, 192, 192),
    (768, 384, 384),
    (800, 400, 400),
)
L5_C4_MESHES = COMMON_MESHES + ((1024, 512, 512),)
TOP3D_XL_FILTER_RADIUS_ELEMENTS = math.sqrt(3.0)
TOPAX_EQUIVALENT_FILTER_RADIUS_ELEMENTS = math.sqrt(6.0)

OUTPUT_FIELDS = (
    "series",
    "label",
    "implementation",
    "run_id",
    "levels",
    "galerkin_crossover_level",
    "resolution",
    "ex",
    "ey",
    "ez",
    "n_elements",
    "runtime_design_iterations",
    "runtime_s",
    "mean_physics_pcg_iterations",
    "memory_design_iterations",
    "gpu_peak_above_baseline_gib",
)


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    levels: int
    crossover: int
    meshes: tuple[tuple[int, int, int], ...]

    def matches(self, configuration: dict) -> bool:
        return (
            configuration.get("schema_version") == 2
            and configuration.get("benchmark") == "wang_2025"
            and configuration.get("implementation") == "topax"
            and configuration.get("backend") == "gpu"
            and configuration.get("levels") == self.levels
            and configuration.get("galerkin_crossover_level")
            == self.crossover
        )


SERIES = (
    Series("topax_l4_c3", "topax L4-C3", 4, 3, COMMON_MESHES),
    Series("topax_l5_c4", "topax L5-C4", 5, 4, L5_C4_MESHES),
)


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


def mesh_name(mesh: tuple[int, int, int]) -> str:
    return "x".join(map(str, mesh))


def matching_runs(
    results_root: Path,
    series: Series,
) -> dict[tuple[int, int, int], dict[str, Path]]:
    matches: dict[tuple[int, int, int], dict[str, Path]] = {}
    for mesh in series.meshes:
        mesh_root = results_root / mesh_name(mesh)
        if not mesh_root.is_dir():
            raise ValueError(f"missing mesh directory: {mesh_root}")
        mesh_matches = {}
        for run_dir in sorted(mesh_root.iterdir()):
            configuration_path = run_dir / "configuration.toml"
            if not run_dir.is_dir() or not configuration_path.is_file():
                continue
            configuration = read_configuration(configuration_path)
            if series.matches(configuration):
                mesh_matches[run_dir.name] = run_dir
        if not mesh_matches:
            raise ValueError(
                f"no {series.label} result found for {mesh_name(mesh)}"
            )
        matches[mesh] = mesh_matches
    return matches


def select_campaign(
    results_root: Path,
    series: Series,
    requested_run_id: str | None,
) -> tuple[str, dict[tuple[int, int, int], Path]]:
    candidates = matching_runs(results_root, series)
    common_ids = set.intersection(
        *(set(mesh_runs) for mesh_runs in candidates.values())
    )
    if requested_run_id is not None:
        if requested_run_id not in common_ids:
            raise ValueError(
                f"{series.label} run {requested_run_id} does not contain all "
                f"{len(series.meshes)} expected meshes"
            )
        run_id = requested_run_id
    elif len(common_ids) == 1:
        run_id = common_ids.pop()
    elif not common_ids:
        raise ValueError(
            f"no single {series.label} run ID contains all expected meshes"
        )
    else:
        choices = ", ".join(sorted(common_ids))
        raise ValueError(
            f"multiple complete {series.label} runs found ({choices}); "
            f"select one with --{series.key.replace('_', '-')}-run"
        )
    return run_id, {mesh: runs[run_id] for mesh, runs in candidates.items()}


def validate_configuration(
    configuration: dict,
    series: Series,
    mesh: tuple[int, int, int],
    path: Path,
) -> None:
    if not series.matches(configuration):
        raise ValueError(f"unexpected configuration in {path}")
    if tuple(configuration.get("elements", ())) != mesh:
        raise ValueError(f"element dimensions do not match path in {path}")
    if configuration.get("resolution") != str(mesh[0]):
        raise ValueError(f"resolution does not match mesh in {path}")
    if configuration.get("filter_discretization") != (
        "released TOP3D_XL element operator"
    ):
        raise ValueError(f"result does not use the released TOP3D_XL filter: {path}")
    if not math.isclose(
        configuration.get("filter_radius_elements", math.nan),
        TOP3D_XL_FILTER_RADIUS_ELEMENTS,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ):
        raise ValueError(f"unexpected TOP3D_XL filter radius in {path}")
    if not math.isclose(
        configuration.get(
            "equivalent_topax_helmholtz_radius_elements", math.nan
        ),
        TOPAX_EQUIVALENT_FILTER_RADIUS_ELEMENTS,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ):
        raise ValueError(f"unexpected equivalent Helmholtz radius in {path}")

    runtime = configuration.get("measurements", {}).get("runtime", {})
    memory = configuration.get("measurements", {}).get("memory", {})
    if runtime.get("design_iteration_limit") != 50:
        raise ValueError(f"runtime run is not a 50-iteration run: {path}")
    if runtime.get("full_benchmark") is not True:
        raise ValueError(f"runtime run is not marked complete: {path}")
    if memory.get("design_iteration_limit") != 10:
        raise ValueError(f"memory run is not a 10-iteration run: {path}")


def evaluate_run(
    series: Series,
    run_id: str,
    mesh: tuple[int, int, int],
    run_dir: Path,
) -> dict[str, str | int | float]:
    configuration_path = run_dir / "configuration.toml"
    configuration = read_configuration(configuration_path)
    validate_configuration(configuration, series, mesh, configuration_path)

    runtime_path = run_dir / "runtime.csv"
    runtime_rows = read_csv(runtime_path)
    iterations = [int(number(row, "iteration", runtime_path)) for row in runtime_rows]
    if iterations != list(range(1, 51)):
        raise ValueError(f"runtime does not contain iterations 1--50: {runtime_path}")
    iteration_times = [
        number(row, "iteration_time_s", runtime_path) for row in runtime_rows
    ]
    pcg_iterations = [
        number(row, "physics_cg_iterations", runtime_path)
        for row in runtime_rows
    ]
    if min(iteration_times) < 0.0 or min(pcg_iterations) < 0.0:
        raise ValueError(f"negative runtime or PCG count in {runtime_path}")

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

    ex, ey, ez = mesh
    return {
        "series": series.key,
        "label": series.label,
        "implementation": "topax",
        "run_id": run_id,
        "levels": series.levels,
        "galerkin_crossover_level": series.crossover,
        "resolution": ex,
        "ex": ex,
        "ey": ey,
        "ez": ez,
        "n_elements": ex * ey * ez,
        "runtime_design_iterations": len(runtime_rows),
        "runtime_s": math.fsum(iteration_times),
        "mean_physics_pcg_iterations": statistics.fmean(pcg_iterations),
        "memory_design_iterations": memory_iterations,
        "gpu_peak_above_baseline_gib": peak_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Topax Wang GPU runtime, memory, and PCG results."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    for series in SERIES:
        parser.add_argument(
            f"--{series.key.replace('_', '-')}-run",
            dest=f"{series.key}_run",
            help=f"explicit run ID for {series.label}",
        )
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    rows = []
    selected = []
    try:
        for series in SERIES:
            requested = getattr(args, f"{series.key}_run")
            run_id, run_dirs = select_campaign(
                results_root, series, requested
            )
            selected.append((series.label, run_id))
            rows.extend(
                evaluate_run(series, run_id, mesh, run_dirs[mesh])
                for mesh in series.meshes
            )
    except ValueError as error:
        raise SystemExit(str(error)) from None

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for label, run_id in selected:
        print(f"{label}: {run_id}")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
