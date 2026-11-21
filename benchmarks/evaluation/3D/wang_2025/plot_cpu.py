#!/usr/bin/env python3
"""Plot the available Topax and TOP3D_XL Wang CPU results."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.ticker import FuncFormatter


EVALUATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_DIR))
from plot_style import WANG_SERIES_STYLES


BENCHMARKS_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "wang_cpu.csv"
RESOLUTIONS = (96, 192, 384, 768, 800)
EXCLUDED_RESOLUTIONS = {768}
ELEMENT_LIMITS = (
    (96 * 48 * 48) / 1.7,
    (800 * 400 * 400) * 1.25,
)
SERIES = ("topax_l4_c3", "topax_l5_c4", "top3d_xl")


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
        "resolution",
        "n_elements",
        "runtime_s",
        "mean_physics_pcg_iterations",
        "host_peak_rss_gib",
    }
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"missing evaluation fields: {', '.join(sorted(missing))}")

    by_series: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["series"] not in SERIES:
            raise ValueError(f"unexpected series: {row['series']}")
        by_series.setdefault(row["series"], []).append(row)
    for key in SERIES:
        series_rows = by_series.get(key, [])
        if not series_rows:
            raise ValueError(f"no evaluated rows for {key}")
        series_rows.sort(key=lambda row: int(row["resolution"]))
        resolutions = [int(row["resolution"]) for row in series_rows]
        if len(resolutions) != len(set(resolutions)):
            raise ValueError(f"duplicate resolution in {key}")
        if any(resolution not in RESOLUTIONS for resolution in resolutions):
            raise ValueError(f"unexpected resolution in {key}: {resolutions}")
        for row in series_rows:
            for field in (
                "runtime_s",
                "mean_physics_pcg_iterations",
                "host_peak_rss_gib",
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


def annotate_runtime_speedups(ax, data) -> None:
    rows_by_series = {
        key: {int(row["resolution"]): row for row in data[key]} for key in SERIES
    }
    common_resolutions = (
        set(rows_by_series["topax_l4_c3"])
        & set(rows_by_series["topax_l5_c4"])
        & set(rows_by_series["top3d_xl"]) - EXCLUDED_RESOLUTIONS
    )
    for resolution in sorted(common_resolutions):
        l4_row = rows_by_series["topax_l4_c3"][resolution]
        l5_row = rows_by_series["topax_l5_c4"][resolution]
        competitor_row = rows_by_series["top3d_xl"][resolution]
        l4_time = float(l4_row["runtime_s"])
        l5_time = float(l5_row["runtime_s"])
        competitor_time = float(competitor_row["runtime_s"])
        if min(l4_time, l5_time, competitor_time) <= 0.0:
            raise ValueError(f"runtime must be positive at resolution {resolution}")

        text_properties = {"fontsize": 9.0}
        label = HPacker(
            children=(
                TextArea(
                    f"{competitor_time / l5_time:.1f}×",
                    textprops={
                        **text_properties,
                        "color": WANG_SERIES_STYLES["topax_l5_c4"]["color"],
                    },
                ),
                TextArea(" / ", textprops=text_properties),
                TextArea(
                    f"{competitor_time / l4_time:.1f}×",
                    textprops={
                        **text_properties,
                        "color": WANG_SERIES_STYLES["topax_l4_c3"]["color"],
                    },
                ),
            ),
            align="center",
            pad=0.0,
            sep=0.0,
        )
        if resolution == min(RESOLUTIONS):
            horizontal_alignment = 0.5
            horizontal_offset = 2.0
            vertical_offset = 16.0
        else:
            horizontal_alignment = 1.0
            horizontal_offset = -5.0
            vertical_offset = 7.0
        annotation = AnnotationBbox(
            label,
            (
                int(competitor_row["n_elements"]),
                competitor_time,
            ),
            xybox=(horizontal_offset, vertical_offset),
            xycoords="data",
            boxcoords="offset points",
            box_alignment=(horizontal_alignment, 0.0),
            frameon=False,
            pad=0.0,
        )
        ax.add_artist(annotation)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the available Topax and TOP3D_XL Wang CPU results."
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
        ("host_peak_rss_gib", "peak host memory [GiB]", False, "memory"),
        (
            "mean_physics_pcg_iterations",
            "mean MGPCG iterations",
            False,
            "pcg",
        ),
    )
    for field, ylabel, log_y, suffix in specifications:
        fig, ax = plt.subplots(figsize=(5.2, 3.15), constrained_layout=True)
        for key in SERIES:
            rows = [
                row
                for row in data[key]
                if int(row["resolution"]) not in EXCLUDED_RESOLUTIONS
            ]
            if rows:
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
            annotate_runtime_speedups(ax, data)
            ax.legend(loc="best")
        save(fig, output_dir / f"wang_cpu_{suffix}.pdf")


if __name__ == "__main__":
    main()
