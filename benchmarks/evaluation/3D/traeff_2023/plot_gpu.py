#!/usr/bin/env python3
"""Plot the evaluated Träff GPU benchmark results."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


EVALUATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_DIR))
from plot_style import IMPLEMENTATION_STYLES, TOPAX_CONFIGURATION_STYLES


BENCHMARKS_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "traeff_gpu.csv"
EXPECTED_ELEMENTS = tuple(
    ex * ey * ez
    for ex, ey, ez in (
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
)
PCG_CAP = 80.0
X_LIMITS = (EXPECTED_ELEMENTS[0] / 1.7, 120_000_000)

IMPLEMENTATION_STYLE_KEYS = {
    "topax_l4_c2": "topax",
    "futtop": "futtop",
    "pyfantom": "pyfantom",
}
EXPECTED_SERIES = (
    "topax_l4_c2",
    "topax_l4_c3",
    "futtop",
    "pyfantom",
)


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 12.5,
            "legend.fontsize": 11.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_evaluation(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.is_file():
        raise ValueError(f"evaluation CSV not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    required = {
        "series",
        "n_elements",
        "runtime_s",
        "mean_physics_pcg_iterations",
        "gpu_peak_above_baseline_gib",
    }
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"missing evaluation fields: {', '.join(sorted(missing))}")

    by_series: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_series.setdefault(row["series"], []).append(row)
    for key in EXPECTED_SERIES:
        series_rows = by_series.get(key, [])
        series_rows.sort(key=lambda row: int(row["n_elements"]))
        elements = tuple(int(row["n_elements"]) for row in series_rows)
        if elements != EXPECTED_ELEMENTS:
            raise ValueError(
                f"{key} does not contain the expected 11 meshes: {elements}"
            )
        for row in series_rows:
            for field in (
                "runtime_s",
                "mean_physics_pcg_iterations",
                "gpu_peak_above_baseline_gib",
            ):
                value = float(row[field])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"invalid {field} for {key}")
    return by_series


def compact_number(value: float, _position=None) -> str:
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    if value >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def polish_axis(ax, *, log_y: bool = False) -> None:
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def elements(rows: list[dict[str, str]]) -> list[int]:
    return [int(row["n_elements"]) for row in rows]


def plot_series(ax, rows, key, field, *, capped=False) -> None:
    y_values = values(rows, field)
    if capped:
        y_values = [min(value, PCG_CAP) for value in y_values]
    style_key = IMPLEMENTATION_STYLE_KEYS[key]
    ax.plot(elements(rows), y_values, **IMPLEMENTATION_STYLES[style_key])


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def implementation_plots(data, output_dir: Path) -> None:
    keys = ("topax_l4_c2", "futtop", "pyfantom")

    fig, ax = plt.subplots(figsize=(5.2, 3.15), constrained_layout=True)
    for key in keys:
        plot_series(ax, data[key], key, "runtime_s")
    polish_axis(ax, log_y=True)
    ax.set_xlim(*X_LIMITS)
    ax.set_xlabel("number of elements")
    ax.set_ylabel("time [s]")
    ax.legend(loc="upper left")

    topax = data["topax_l4_c2"]
    competitors = {
        key: {int(row["n_elements"]): float(row["runtime_s"]) for row in data[key]}
        for key in ("futtop", "pyfantom")
    }
    for index, row in enumerate(topax):
        n_elements = int(row["n_elements"])
        topax_time = float(row["runtime_s"])
        competitor_time = min(
            times[n_elements] for times in competitors.values()
        )
        speedup = competitor_time / topax_time
        above = index % 3 != 1
        ax.annotate(
            f"{speedup:.1f}×",
            (n_elements, topax_time),
            xytext=(0, 7 if above else -7),
            textcoords="offset points",
            ha="center",
            va="bottom" if above else "top",
            fontsize=9.0,
            color=IMPLEMENTATION_STYLES["topax"]["color"],
        )
    save(fig, output_dir / "traeff_gpu_runtime.pdf")

    fig, ax = plt.subplots(figsize=(5.2, 3.15), constrained_layout=True)
    for key in keys:
        plot_series(ax, data[key], key, "gpu_peak_above_baseline_gib")
    polish_axis(ax)
    ax.set_xlim(*X_LIMITS)
    ax.set_xlabel("number of elements")
    ax.set_ylabel("peak device memory [GiB]")
    save(fig, output_dir / "traeff_gpu_memory.pdf")

    fig, ax = plt.subplots(figsize=(5.2, 3.15), constrained_layout=True)
    for key in keys:
        plot_series(
            ax,
            data[key],
            key,
            "mean_physics_pcg_iterations",
            capped=True,
        )
    pyfantom = data["pyfantom"]
    annotation_index = 0
    for row in pyfantom:
        actual = float(row["mean_physics_pcg_iterations"])
        if actual <= PCG_CAP:
            continue
        above = annotation_index % 2 == 0
        ax.annotate(
            f"{actual:.1f}",
            (int(row["n_elements"]), PCG_CAP),
            xytext=(0, 5 if above else -6),
            textcoords="offset points",
            ha="center",
            va="bottom" if above else "top",
            fontsize=9.0,
            color=IMPLEMENTATION_STYLES["pyfantom"]["color"],
            annotation_clip=False,
        )
        annotation_index += 1
    polish_axis(ax)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(0.0, 86.0)
    ax.set_xlabel("number of elements")
    ax.set_ylabel("mean MGPCG iterations")
    save(fig, output_dir / "traeff_gpu_pcg.pdf")


def topax_configuration_plots(data, output_dir: Path) -> None:
    keys = ("topax_l4_c2", "topax_l4_c3")

    specifications = (
        ("runtime_s", "time [s]", True, "runtime"),
        (
            "gpu_peak_above_baseline_gib",
            "peak device memory [GiB]",
            False,
            "memory",
        ),
        (
            "mean_physics_pcg_iterations",
            "mean MGPCG iterations",
            False,
            "pcg",
        ),
    )
    for field, ylabel, log_y, suffix in specifications:
        fig, ax = plt.subplots(figsize=(5.2, 3.15), constrained_layout=True)
        for key in keys:
            ax.plot(
                elements(data[key]),
                values(data[key], field),
                **TOPAX_CONFIGURATION_STYLES[key],
            )
        polish_axis(ax, log_y=log_y)
        ax.set_xlim(*X_LIMITS)
        if suffix == "pcg":
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel("number of elements")
        ax.set_ylabel(ylabel)
        if suffix == "runtime":
            ax.legend(loc="best")
        save(fig, output_dir / f"traeff_gpu_topax_{suffix}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot both Träff GPU comparison figure sets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        data = read_evaluation(args.input.expanduser().resolve())
    except ValueError as error:
        raise SystemExit(str(error)) from None
    output_dir = args.output_dir.expanduser().resolve()
    paper_style()
    implementation_plots(data, output_dir)
    topax_configuration_plots(data, output_dir)


if __name__ == "__main__":
    main()
