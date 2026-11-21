#!/usr/bin/env python3
"""Render a saved design from the filtered Träff R640 run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


DEFAULT_ISOVALUE = 0.5
DEFAULT_DOMAIN_SIZE = (2.0, 1.0, 1.0)

# The Wang camera scaled to the twice-as-large Träff domain. Use --show to
# refine it for the final figure; the selected parameters are printed on exit.
CAMERA_POSITION = (
    4.056536036920479,
    -1.114970953333646,
    1.4197258243746502,
)
CAMERA_FOCAL_POINT = (
    1.0033661900566442,
    0.5138951915677205,
    0.4790085617525537,
)
CAMERA_VIEW_UP = (
    -0.1848947160398711,
    0.2081280953800152,
    0.9604668864120318,
)
CAMERA_PARALLEL_SCALE = 0.8070842541645029


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="completed R640 visualization-run directory",
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
        raise ValueError(
            f"expected a three-dimensional density field, got {density.shape}"
        )
    if not 0.0 < isovalue < 1.0:
        raise ValueError(
            f"the density isovalue must lie in (0, 1), got {isovalue}"
        )

    # VTK stores the x index contiguously, whereas these arrays use (x, y, z)
    # with z contiguous. Reverse VTK's axes to avoid a full transposed copy,
    # then transform the extracted coordinates back to (x, y, z).
    elements = np.asarray(density.shape, dtype=np.int64)
    spacing = np.asarray(domain_size, dtype=np.float64) / elements

    # The exterior void layer closes material surfaces at the domain boundary
    # without modifying any density computed by the optimization.
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
    plotter.enable_anti_aliasing("ssaa")

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
    density_path = (
        run_directory / "density" / f"iteration_{args.iteration:03d}.npy"
    )
    if not density_path.is_file():
        raise FileNotFoundError(f"density field not found: {density_path}")

    destination = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_directory / "renders" / f"iteration_{args.iteration:03d}.png"
    )
    density = np.load(density_path, mmap_mode="r", allow_pickle=False)
    print(
        f"extracting the rho={args.isovalue:g} material boundary from "
        f"{density_path} with shape {density.shape}",
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
