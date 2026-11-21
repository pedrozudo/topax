#!/usr/bin/env python3
"""Create the Topax/OpenMP Träff steady-memory comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


BENCHMARKS_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = BENCHMARKS_DIR / "results" / "traeff_2023"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "traeff_memory_comparison.csv"
)

MESHES = (
    (64, 32, 32),
    (80, 40, 40),
    (96, 48, 48),
    (128, 64, 64),
    (160, 80, 80),
    (192, 96, 96),
    (256, 128, 128),
    (320, 160, 160),
    (400, 200, 200),
    (512, 256, 256),
    (640, 320, 320),
)

OUTPUT_FIELDS = (
    "ex",
    "ey",
    "ez",
    "n_elements",
    "topax_run_id",
    "openmp_run_id",
    "topax_setup_host_peak_gib",
    "topax_setup_hbm_peak_gib",
    "topax_steady_host_median_gib",
    "topax_steady_host_p95_gib",
    "topax_steady_hbm_median_gib",
    "topax_steady_hbm_p95_gib",
    "topax_steady_combined_median_gib",
    "topax_steady_combined_p95_gib",
    "topax_release_steady_host_median_gib",
    "topax_release_steady_host_p95_gib",
    "topax_release_steady_hbm_median_gib",
    "topax_release_steady_hbm_p95_gib",
    "topax_release_steady_combined_median_gib",
    "topax_release_steady_combined_p95_gib",
    "openmp_steady_host_median_gib",
    "openmp_steady_host_p95_gib",
    "openmp_steady_hbm_median_gib",
    "openmp_steady_hbm_p95_gib",
    "openmp_steady_combined_median_gib",
    "openmp_steady_combined_p95_gib",
    "openmp_measurement_window_terminated",
)


def mesh_name(mesh: tuple[int, int, int]) -> str:
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


def validate_combined(
    row: dict[str, str],
    path: Path,
    *,
    host_field: str,
    gpu_field: str,
    combined_field: str,
) -> None:
    host = number(row, host_field, path)
    gpu = number(row, gpu_field, path)
    combined = number(row, combined_field, path)
    if min(host, gpu, combined) < 0.0:
        raise ValueError(f"negative memory value in {path}")
    if not math.isclose(combined, host + gpu, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"inconsistent {combined_field!r} in {path}")


def matches_topax(configuration: dict, mesh: tuple[int, int, int]) -> bool:
    measurement = configuration.get("measurements", {}).get(
        "steady_memory", {}
    )
    return (
        configuration.get("schema_version") == 2
        and configuration.get("benchmark") == "traeff_2023"
        and configuration.get("implementation") == "topax"
        and configuration.get("backend") == "gpu"
        and tuple(configuration.get("elements", ())) == mesh
        and configuration.get("levels") == 4
        and configuration.get("galerkin_crossover_level") == 2
        and configuration.get("precision") == "float64"
        and measurement.get("design_iteration_limit") == 20
        and measurement.get("full_benchmark") is True
        and measurement.get("jax_preallocate") is False
    )


def matches_openmp(configuration: dict, mesh: tuple[int, int, int]) -> bool:
    measurement = configuration.get("measurements", {}).get("memory", {})
    return (
        configuration.get("schema_version") == 2
        and configuration.get("benchmark") == "traeff_2023"
        and configuration.get("implementation") == "openmp"
        and tuple(configuration.get("elements", ())) == mesh
        and configuration.get("levels") == 4
        and measurement.get("design_iteration_limit") == 20
        and measurement.get("bounded_measurement") is True
        and measurement.get("steady_state_delay_seconds") == 30
    )


def matching_runs(
    results_root: Path,
    implementation: str,
    result_name: str,
    matcher,
) -> dict[tuple[int, int, int], dict[str, Path]]:
    root = results_root / implementation / "gpu"
    if not root.is_dir():
        raise ValueError(f"missing result directory: {root}")

    matches = {}
    for mesh in MESHES:
        mesh_root = root / mesh_name(mesh)
        if not mesh_root.is_dir():
            raise ValueError(f"missing mesh directory: {mesh_root}")
        mesh_matches = {}
        for run_dir in sorted(mesh_root.iterdir()):
            configuration_path = run_dir / "configuration.toml"
            result_path = run_dir / result_name
            if (
                run_dir.is_dir()
                and configuration_path.is_file()
                and result_path.is_file()
                and matcher(read_configuration(configuration_path), mesh)
            ):
                mesh_matches[run_dir.name] = run_dir
        if not mesh_matches:
            raise ValueError(
                f"no matching {implementation} result found for "
                f"{mesh_name(mesh)}"
            )
        matches[mesh] = mesh_matches
    return matches


def select_campaign(
    matches: dict[tuple[int, int, int], dict[str, Path]],
    label: str,
    requested_run_id: str | None,
) -> tuple[str, dict[tuple[int, int, int], Path]]:
    common_ids = set.intersection(
        *(set(mesh_runs) for mesh_runs in matches.values())
    )
    if requested_run_id is not None:
        if requested_run_id not in common_ids:
            raise ValueError(
                f"{label} run {requested_run_id} does not contain all 11 "
                "meshes"
            )
        run_id = requested_run_id
    elif len(common_ids) == 1:
        run_id = common_ids.pop()
    elif not common_ids:
        raise ValueError(f"no single {label} run contains all 11 meshes")
    else:
        choices = ", ".join(sorted(common_ids))
        option = label.lower().replace(" ", "-")
        raise ValueError(
            f"multiple complete {label} runs found ({choices}); select one "
            f"with --{option}-run"
        )
    return run_id, {mesh: runs[run_id] for mesh, runs in matches.items()}


def evaluate_topax(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    if len(rows) != 2:
        raise ValueError(f"expected two Topax variants in {path}")
    by_variant = {row.get("variant", ""): row for row in rows}
    expected = {"natural", "release_initialization_memory"}
    if set(by_variant) != expected:
        raise ValueError(f"unexpected Topax variants in {path}")

    for variant, expected_release in (
        ("natural", 0),
        ("release_initialization_memory", 1),
    ):
        row = by_variant[variant]
        if int(number(row, "release_initialization_memory", path)) != expected_release:
            raise ValueError(f"incorrect cleanup flag for {variant} in {path}")
        if int(number(row, "design_iterations_completed", path)) != 20:
            raise ValueError(f"{variant} did not complete 20 iterations: {path}")
        if int(number(row, "application_exit_status", path)) != 0:
            raise ValueError(f"{variant} failed: {path}")
        if number(row, "host_steady_samples", path) <= 0:
            raise ValueError(f"{variant} has no host samples: {path}")
        if number(row, "gpu_steady_samples", path) <= 0:
            raise ValueError(f"{variant} has no GPU samples: {path}")
        for statistic in ("median", "p95"):
            validate_combined(
                row,
                path,
                host_field=f"host_steady_rss_{statistic}_gib",
                gpu_field=f"gpu_steady_delta_{statistic}_gib",
                combined_field=f"combined_steady_{statistic}_gib",
            )
    return by_variant


def evaluate_openmp(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"expected one OpenMP memory row in {path}")
    row = rows[0]
    terminated = int(number(row, "measurement_window_terminated", path))
    status = int(number(row, "application_exit_status", path))
    if terminated not in (0, 1) or status != (143 if terminated else 0):
        raise ValueError(f"inconsistent OpenMP termination status in {path}")
    if number(row, "host_steady_samples", path) <= 0:
        raise ValueError(f"OpenMP run has no host samples: {path}")
    if number(row, "gpu_steady_samples", path) <= 0:
        raise ValueError(f"OpenMP run has no GPU samples: {path}")
    for statistic in ("median", "p95"):
        validate_combined(
            row,
            path,
            host_field=f"host_steady_rss_{statistic}_gib",
            gpu_field=f"gpu_steady_delta_{statistic}_gib",
            combined_field=f"combined_steady_{statistic}_gib",
        )
    return row


def output_row(
    mesh: tuple[int, int, int],
    topax_run_id: str,
    openmp_run_id: str,
    topax_path: Path,
    openmp_path: Path,
) -> dict[str, int | float | str]:
    topax = evaluate_topax(topax_path)
    natural = topax["natural"]
    released = topax["release_initialization_memory"]
    openmp = evaluate_openmp(openmp_path)
    ex, ey, ez = mesh

    def topax_value(row, field):
        return number(row, field, topax_path)

    def openmp_value(field):
        return number(openmp, field, openmp_path)

    return {
        "ex": ex,
        "ey": ey,
        "ez": ez,
        "n_elements": ex * ey * ez,
        "topax_run_id": topax_run_id,
        "openmp_run_id": openmp_run_id,
        "topax_setup_host_peak_gib": topax_value(
            natural, "host_setup_peak_rss_gib"
        ),
        "topax_setup_hbm_peak_gib": topax_value(
            natural, "gpu_setup_peak_delta_gib"
        ),
        "topax_steady_host_median_gib": topax_value(
            natural, "host_steady_rss_median_gib"
        ),
        "topax_steady_host_p95_gib": topax_value(
            natural, "host_steady_rss_p95_gib"
        ),
        "topax_steady_hbm_median_gib": topax_value(
            natural, "gpu_steady_delta_median_gib"
        ),
        "topax_steady_hbm_p95_gib": topax_value(
            natural, "gpu_steady_delta_p95_gib"
        ),
        "topax_steady_combined_median_gib": topax_value(
            natural, "combined_steady_median_gib"
        ),
        "topax_steady_combined_p95_gib": topax_value(
            natural, "combined_steady_p95_gib"
        ),
        "topax_release_steady_host_median_gib": topax_value(
            released, "host_steady_rss_median_gib"
        ),
        "topax_release_steady_host_p95_gib": topax_value(
            released, "host_steady_rss_p95_gib"
        ),
        "topax_release_steady_hbm_median_gib": topax_value(
            released, "gpu_steady_delta_median_gib"
        ),
        "topax_release_steady_hbm_p95_gib": topax_value(
            released, "gpu_steady_delta_p95_gib"
        ),
        "topax_release_steady_combined_median_gib": topax_value(
            released, "combined_steady_median_gib"
        ),
        "topax_release_steady_combined_p95_gib": topax_value(
            released, "combined_steady_p95_gib"
        ),
        "openmp_steady_host_median_gib": openmp_value(
            "host_steady_rss_median_gib"
        ),
        "openmp_steady_host_p95_gib": openmp_value(
            "host_steady_rss_p95_gib"
        ),
        "openmp_steady_hbm_median_gib": openmp_value(
            "gpu_steady_delta_median_gib"
        ),
        "openmp_steady_hbm_p95_gib": openmp_value(
            "gpu_steady_delta_p95_gib"
        ),
        "openmp_steady_combined_median_gib": openmp_value(
            "combined_steady_median_gib"
        ),
        "openmp_steady_combined_p95_gib": openmp_value(
            "combined_steady_p95_gib"
        ),
        "openmp_measurement_window_terminated": int(
            openmp_value("measurement_window_terminated")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Topax/OpenMP Träff steady-memory results."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topax-run", help="explicit Topax steady-memory run ID")
    parser.add_argument("--openmp-run", help="explicit OpenMP memory run ID")
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    try:
        topax_matches = matching_runs(
            results_root,
            "topax",
            "steady_memory.csv",
            matches_topax,
        )
        openmp_matches = matching_runs(
            results_root,
            "openmp",
            "memory.csv",
            matches_openmp,
        )
        topax_run_id, topax_dirs = select_campaign(
            topax_matches, "Topax", args.topax_run
        )
        openmp_run_id, openmp_dirs = select_campaign(
            openmp_matches, "OpenMP", args.openmp_run
        )
        rows = [
            output_row(
                mesh,
                topax_run_id,
                openmp_run_id,
                topax_dirs[mesh] / "steady_memory.csv",
                openmp_dirs[mesh] / "memory.csv",
            )
            for mesh in MESHES
        ]
    except ValueError as error:
        raise SystemExit(str(error)) from None

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"topax: {topax_run_id}")
    print(f"OpenMP: {openmp_run_id}")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
