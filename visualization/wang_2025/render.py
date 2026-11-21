#!/usr/bin/env python3
"""Render the final design from a saved Wang R800 density field."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


DEFAULT_ISOVALUE = 0.5
DEFAULT_DOMAIN_SIZE = (1.0, 0.5, 0.5)
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a density isosurface from the Wang R800 run."
    )
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="completed R800 visualization-run directory",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=50,
        help="saved design iteration to render (default: 50)",
    )
    parser.add_argument(
        "--isovalue",
        type=float,
        default=DEFAULT_ISOVALUE,
        help="density value defining the material boundary (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG destination (default: <run>/renders/iteration_NNN.png)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1920, 1080),
        help="rendered image size in pixels (default: 1920 1080)",
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
        "--show",
        action="store_true",
        help="open an interactive window after constructing the surface",
    )
    parser.add_argument(
        "--axes",
        action="store_true",
        help="include an orientation triad",
    )
    return parser.parse_args()


def construct_surface(density: np.ndarray, isovalue: float, domain_size):
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError(
            "PyVista is required for rendering. Install it with "
            "`python -m pip install pyvista`."
        ) from error

    if density.ndim != 3:
        raise ValueError(f"expected a three-dimensional density field, got {density.shape}")
    if not 0.0 < isovalue < 1.0:
        raise ValueError(f"the density isovalue must lie in (0, 1), got {isovalue}")

    # NumPy stores the z index contiguously for an (x, y, z) array, whereas
    # VTK stores the x index contiguously. Reversing the VTK axis convention
    # lets it consume the C-contiguous field without making a Fortran-ordered
    # copy. The extracted surface is transformed back to physical (x, y, z)
    # coordinates below.
    elements = np.asarray(density.shape, dtype=np.int64)
    spacing = np.asarray(domain_size, dtype=np.float64) / elements

    # A one-cell void layer closes surfaces where material reaches the
    # design-domain boundary without changing any computed density value.
    padded_shape = tuple(int(size) + 2 for size in density.shape)
    scalar_field = np.zeros(padded_shape, dtype=density.dtype)
    scalar_field[1:-1, 1:-1, 1:-1] = density
    origin = -spacing / 2.0

    image = pv.ImageData()
    image.dimensions = tuple(int(value) for value in scalar_field.shape[::-1])
    image.spacing = tuple(float(value) for value in spacing[::-1])
    image.origin = tuple(float(value) for value in origin[::-1])
    image.point_data["density"] = scalar_field.reshape(-1)

    surface = image.contour(
        isosurfaces=[isovalue],
        scalars="density",
        method="flying_edges",
    )
    if surface.n_points == 0:
        raise RuntimeError(
            f"the density field contains no {isovalue:g} isosurface"
        )

    surface.points = surface.points[:, (2, 1, 0)]
    return surface


def render(surface, destination: Path, args: argparse.Namespace) -> None:
    import pyvista as pv

    destination.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(
        off_screen=not args.show,
        window_size=tuple(args.window_size),
    )
    plotter.set_background(args.background)
    plotter.add_mesh(
        surface,
        color=args.color,
        smooth_shading=True,
        split_sharp_edges=False,
        show_scalar_bar=False,
    )
    if args.axes:
        plotter.add_axes()

    plotter.camera.position = CAMERA_POSITION
    plotter.camera.focal_point = CAMERA_FOCAL_POINT
    plotter.camera.up = CAMERA_VIEW_UP
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = CAMERA_PARALLEL_SCALE

    if args.show:
        plotter.show(auto_close=False)
        plotter.screenshot(str(destination), return_img=False)
        position = tuple(float(value) for value in plotter.camera.position)
        focal_point = tuple(float(value) for value in plotter.camera.focal_point)
        view_up = tuple(float(value) for value in plotter.camera.up)
        parallel_scale = float(plotter.camera.parallel_scale)
        print("camera parameters:")
        print(f"  position = {position}")
        print(f"  focal_point = {focal_point}")
        print(f"  view_up = {view_up}")
        print(f"  parallel_scale = {parallel_scale}")
        plotter.close()
    else:
        plotter.screenshot(str(destination), return_img=False)
        plotter.close()


def main() -> None:
    args = parse_arguments()
    run_directory = args.run_directory.expanduser().resolve()
    density_path = run_directory / "density" / f"iteration_{args.iteration:03d}.npy"
    if not density_path.is_file():
        raise FileNotFoundError(f"density field not found: {density_path}")

    destination = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_directory / "renders" / f"iteration_{args.iteration:03d}.png"
    )
    density = np.load(density_path, mmap_mode="r")
    print(
        f"extracting the rho={args.isovalue:g} material boundary from "
        f"{density_path} "
        f"with shape {density.shape}",
        flush=True,
    )
    surface = construct_surface(density, args.isovalue, DEFAULT_DOMAIN_SIZE)
    print(
        f"surface: {surface.n_points:,} points, {surface.n_cells:,} triangles",
        flush=True,
    )
    render(surface, destination, args)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
