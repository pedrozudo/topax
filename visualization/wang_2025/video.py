#!/usr/bin/env python3
"""Render the Wang R800 density history as a sequence of PNG frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np

from render import DEFAULT_DOMAIN_SIZE, construct_surface


CAMERA_POSITION = (
    1.9721823851804094,
    -0.7098221133795932,
    0.5829983377479097,
)
CAMERA_FOCAL_POINT = (
    0.5016830950283221,
    0.2569475957838602,
    0.23950428087627684,
)
CAMERA_VIEW_UP = (
    -0.1384828388155309,
    0.13781532202849522,
    0.9807290351405806,
)
CAMERA_PARALLEL_SCALE = 0.40354212708225146
FRAME_PATTERN = re.compile(r"iteration_(\d+)\.npy")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="downloaded run containing the original float32 density directory",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="directory in which to write frame_NNN.png images",
    )
    parser.add_argument(
        "--isovalue",
        type=float,
        default=0.5,
        help="density value defining the material boundary (default: 0.5)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1920, 1080),
        help="rendered frame size in pixels (default: 1920 1080)",
    )
    parser.add_argument(
        "--color",
        default="#d6d6d6",
        help="surface color understood by PyVista (default: #d6d6d6)",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="background color understood by PyVista (default: white)",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="omit the design-iteration label",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing PNG frames",
    )
    return parser.parse_args()


def discover_frames(directory: Path) -> list[tuple[int, Path]]:
    frames: list[tuple[int, Path]] = []
    for path in directory.glob("iteration_*.npy"):
        match = FRAME_PATTERN.fullmatch(path.name)
        if match is not None:
            frames.append((int(match.group(1)), path))
    frames.sort(key=lambda item: item[0])

    if len(frames) != 51:
        raise ValueError(f"expected 51 density frames in {directory}, found {len(frames)}")
    indices = [index for index, _ in frames]
    if indices != list(range(51)):
        raise ValueError(f"expected frame indices 0 through 50, got {indices}")
    return frames


def render_frame(
    density_path: Path,
    iteration: int,
    destination: Path,
    args: argparse.Namespace,
) -> None:
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError(
            "PyVista is required. Install it with `python -m pip install pyvista`."
        ) from error

    density = np.load(density_path, mmap_mode="r", allow_pickle=False)
    if density.ndim != 3:
        raise ValueError(f"expected a 3D density in {density_path}, got {density.shape}")
    if not np.issubdtype(density.dtype, np.floating):
        raise ValueError(
            f"{density_path} has dtype {density.dtype}; use the original float density "
            "fields for smooth video surfaces"
        )

    surface = None
    try:
        surface = construct_surface(
            density,
            args.isovalue,
            DEFAULT_DOMAIN_SIZE,
        )
    except RuntimeError as error:
        if "contains no" not in str(error):
            raise

    plotter = pv.Plotter(off_screen=True, window_size=tuple(args.window_size))
    plotter.set_background(args.background)
    if surface is not None:
        plotter.add_mesh(
            surface,
            color=args.color,
            smooth_shading=True,
            split_sharp_edges=False,
            show_scalar_bar=False,
        )
    if not args.no_label:
        plotter.add_text(
            f"Design iteration {iteration}",
            position="upper_left",
            font_size=18,
            color="black",
        )

    plotter.camera.position = CAMERA_POSITION
    plotter.camera.focal_point = CAMERA_FOCAL_POINT
    plotter.camera.up = CAMERA_VIEW_UP
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = CAMERA_PARALLEL_SCALE
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(str(destination), return_img=False)
    plotter.close()


def main() -> None:
    args = parse_arguments()
    if not 0.0 < args.isovalue < 1.0:
        raise ValueError(f"--isovalue must lie in (0, 1), got {args.isovalue}")
    run_directory = args.run_directory.expanduser().resolve()
    density_directory = run_directory / "density"
    if not density_directory.is_dir():
        raise FileNotFoundError(f"density directory not found: {density_directory}")
    frames = discover_frames(density_directory)

    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output_directory.glob("frame_*.png"))
    if existing_frames and not args.overwrite:
        raise FileExistsError(
            f"{output_directory} already contains rendered frames; use --overwrite"
        )

    for number, (iteration, density_path) in enumerate(frames):
        destination = output_directory / f"frame_{number:03d}.png"
        print(
            f"[{number + 1:02d}/{len(frames)}] rendering design iteration "
            f"{iteration}",
            flush=True,
        )
        render_frame(density_path, iteration, destination, args)

    print(f"wrote {len(frames)} frames to {output_directory}")


if __name__ == "__main__":
    try:
        main()
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
