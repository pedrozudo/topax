#!/usr/bin/env python3
"""Plot the evaluated Topax Wang GPU benchmark results."""

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
from plot_style import WANG_SERIES_STYLES


BENCHMARKS_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "wang_gpu.csv"
RESOLUTIONS = (96, 192, 384, 768, 800, 1024)
EXCLUDED_RESOLUTIONS = {768}
ELEMENT_LIMITS = (
    (96 * 48 * 48) / 1.7,
    (1024 * 512 * 512) * 1.25,
)
EXPECTED = {
    "topax_l4_c3": RESOLUTIONS[:-1],
    "topax_l5_c4": RESOLUTIONS,
}


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 12.5,
            "legend.fontsize": 11.0,
            "xtick.labelsize": 10.5,
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
        "resolution",
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
    for key, expected_resolutions in EXPECTED.items():
        series_rows = by_series.get(key, [])
        series_rows.sort(key=lambda row: int(row["resolution"]))
        resolutions = tuple(int(row["resolution"]) for row in series_rows)
        if resolutions != expected_resolutions:
            raise ValueError(
                f"{key} has resolutions {resolutions}; expected "
                f"{expected_resolutions}"
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


def elements(rows: list[dict[str, str]]) -> list[int]:
    return [int(row["n_elements"]) for row in rows]


def values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def compact_number(value: float, _position=None) -> str:
    if value >= 1e9:
        return f"{value / 1e9:g}B"
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
    ax.set_xlim(*ELEMENT_LIMITS)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the Topax Wang L4-C3 and L5-C4 GPU comparison."
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
        for key in ("topax_l4_c3", "topax_l5_c4"):
            rows = [
                row
                for row in data[key]
                if int(row["resolution"]) not in EXCLUDED_RESOLUTIONS
            ]
            ax.plot(
                elements(rows),
                values(rows, field),
                **WANG_SERIES_STYLES[key],
            )
        polish_axis(ax, log_y=log_y)
        if suffix == "pcg":
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel("number of elements", labelpad=8)
        ax.set_ylabel(ylabel)
        if suffix == "runtime":
            ax.legend(loc="best")
        save(fig, output_dir / f"wang_gpu_{suffix}.pdf")


if __name__ == "__main__":
    main()
