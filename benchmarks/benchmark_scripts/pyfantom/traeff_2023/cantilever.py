#!/usr/bin/env python3
"""Träff et al. three-dimensional cantilever benchmark for pyFANTOM."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cupy as cp
import numpy as np

from pyFANTOM.CUDA import (
    FiniteElement,
    MinimumCompliance,
    MultiGrid,
    StructuredFilter3D,
    StructuredMesh3D,
    StructuredStiffnessKernel,
)
from pyFANTOM.Physics import LinearElasticity


ELEMENT_SIZE = 0.5
VOLUME_FRACTION = 0.10
FILTER_RADIUS_ELEMENTS = 1.5
SIMP_PENALTY = 3.0
VOID_STIFFNESS = 1.0e-6
YOUNGS_MODULUS = 1.0
POISSON_RATIO = 0.3
MOVE_LIMIT = 0.2
CG_TOLERANCE = 1.0e-5
CG_MAX_ITERATIONS = 1_000
COARSENINGS = 3
SMOOTHING_STEPS = 3
JACOBI_DAMPING = 0.5
MINIMUM_JACOBI_DAMPING = 0.4
W_CYCLE_LEVEL = 1

RUNTIME_FIELDS = (
    "iteration",
    "compliance",
    "mean_design_density",
    "mean_physical_density",
    "volume_constraint",
    "non_discreteness_reported",
    "change",
    "physics_cg_iterations",
    "physics_relative_residual",
    "filter_cg_iterations",
    "filter_relative_residual",
    "iteration_time_s",
)


def synchronize() -> None:
    cp.cuda.Stream.null.synchronize()


def scalar(value) -> float:
    return float(cp.asnumpy(value).item())


def node_id(i, j, k, nx_nodes, nz_nodes):
    return k + nz_nodes * i + nz_nodes * nx_nodes * j


def support_node_ids(ex: int, ey: int, ez: int) -> cp.ndarray:
    if ey % 4:
        raise ValueError("The support definition requires ey divisible by four.")

    lower = np.arange(0, ey // 4 + 1, dtype=np.int64)
    upper = np.arange(3 * ey // 4, ey + 1, dtype=np.int64)
    y_nodes = np.unique(np.concatenate((lower, upper)))
    z_nodes = np.arange(ez + 1, dtype=np.int64)
    jj, kk = np.meshgrid(y_nodes, z_nodes, indexing="ij")
    ids = node_id(0, jj.ravel(), kk.ravel(), ex + 1, ez + 1)
    return cp.asarray(ids, dtype=cp.int32)


def traeff_load(ex: int, ey: int, ez: int):
    element_x = np.arange(ex, dtype=np.float64)[:, None]
    element_y = np.arange(ey, dtype=np.float64)[None, :]
    radius = (ey + 1.0) / 5.0
    patch = (
        (element_x - float(ex)) ** 2
        + (element_y - (ey - 1.0) / 2.0) ** 2
        < radius**2
    )
    n_patch_elements = int(np.count_nonzero(patch))
    if n_patch_elements == 0:
        raise RuntimeError("The load patch contains no elements.")

    contribution = -patch.astype(np.float64) / (4.0 * n_patch_elements)
    face_force = np.zeros((ex + 1, ey + 1), dtype=np.float64)
    face_force[:-1, :-1] += contribution
    face_force[1:, :-1] += contribution
    face_force[:-1, 1:] += contribution
    face_force[1:, 1:] += contribution

    ii, jj = np.nonzero(face_force)
    ids = node_id(ii, jj, 0, ex + 1, ez + 1)
    forces = np.zeros((ids.size, 3), dtype=np.float64)
    forces[:, 2] = face_force[ii, jj]
    return (
        cp.asarray(ids, dtype=cp.int32),
        cp.asarray(forces),
        n_patch_elements,
        float(forces[:, 2].sum()),
    )


def oc_update(problem: MinimumCompliance, move: float) -> float:
    x = problem.get_desvars()
    dg = problem.nabla_g()
    df = problem.nabla_f()
    oc_parameter = x * cp.nan_to_num(cp.sqrt(cp.maximum(-df / dg, 0.0)), nan=0.0)
    if scalar(cp.abs(oc_parameter).sum()) == 0.0:
        oc_parameter = cp.full_like(oc_parameter, 1.0e-3)

    lower = cp.clip(x - move, 0.0, 1.0)
    upper = cp.clip(x + move, 0.0, 1.0)
    lambda_lower = cp.asarray([1.0e-9], dtype=x.dtype)
    lambda_upper = cp.asarray([1.0e9], dtype=x.dtype)

    while scalar((lambda_upper - lambda_lower) / (lambda_upper + lambda_lower)) > 1.0e-8:
        multiplier = 0.5 * (lambda_lower + lambda_upper)
        candidate = cp.maximum(
            0.0,
            cp.maximum(
                lower,
                cp.minimum(1.0, cp.minimum(upper, oc_parameter / multiplier)),
            ),
        )
        if scalar(problem.g(candidate)) <= 0.0:
            lambda_upper = multiplier
        else:
            lambda_lower = multiplier

    change = scalar(cp.max(cp.abs(candidate - x)))
    problem.desvars = candidate
    problem.iteration += 1
    return change


def non_discreteness(density: cp.ndarray) -> float:
    return scalar(100.0 * cp.mean(4.0 * density * (1.0 - density)))


def write_runtime_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUNTIME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, nargs=3, metavar=("EX", "EY", "EZ"), required=True)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--runtime-csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ex, ey, ez = args.elements
    if min(ex, ey, ez, args.max_iterations) <= 0:
        raise ValueError("Mesh dimensions and iteration count must be positive.")
    divisor = 2**COARSENINGS
    if any(elements % divisor for elements in (ex, ey, ez)):
        raise ValueError(
            f"Mesh {(ex, ey, ez)} must be divisible by {divisor} for "
            f"{COARSENINGS} coarsenings."
        )

    setup_start = time.perf_counter()
    physics = LinearElasticity(E=YOUNGS_MODULUS, nu=POISSON_RATIO)
    mesh = StructuredMesh3D(
        nx=ex,
        ny=ey,
        nz=ez,
        lx=ELEMENT_SIZE * ex,
        ly=ELEMENT_SIZE * ey,
        lz=ELEMENT_SIZE * ez,
        dtype=np.float64,
        physics=physics,
    )
    kernel = StructuredStiffnessKernel(mesh=mesh)
    solver = MultiGrid(
        mesh=mesh,
        kernel=kernel,
        maxiter=CG_MAX_ITERATIONS,
        tol=CG_TOLERANCE,
        n_smooth=SMOOTHING_STEPS,
        omega=JACOBI_DAMPING,
        n_level=COARSENINGS,
        cycle="W",
        w_level=W_CYCLE_LEVEL,
        coarse_solver="cholmod",
        matrix_free=False,
        min_omega=MINIMUM_JACOBI_DAMPING,
    )
    finite_element = FiniteElement(mesh=mesh, kernel=kernel, solver=solver)
    finite_element.add_dirichlet_boundary_condition(
        node_ids=support_node_ids(ex, ey, ez), rhs=0.0
    )
    load_ids, load_values, n_patch_elements, total_force = traeff_load(ex, ey, ez)
    finite_element.add_point_forces(node_ids=load_ids, forces=load_values)

    density_filter = StructuredFilter3D(mesh=mesh, r_min=FILTER_RADIUS_ELEMENTS)
    problem = MinimumCompliance(
        FE=finite_element,
        filter=density_filter,
        E_mul=[YOUNGS_MODULUS],
        void=VOID_STIFFNESS,
        penalty=SIMP_PENALTY,
        volume_fraction=[VOLUME_FRACTION],
        heavyside=False,
        simp="interpolated",
        volume_constraint="physical",
    )
    initial_density = cp.full(mesh.elements.shape[0], VOLUME_FRACTION, dtype=cp.float64)
    problem.desvars = initial_density.copy()
    problem.iteration = 0
    synchronize()
    setup_time = time.perf_counter() - setup_start

    warmup_start = time.perf_counter()
    problem._compute()
    synchronize()
    warmup_time = time.perf_counter() - warmup_start
    if not solver.last_converged:
        raise RuntimeError(
            f"Warmup PCG did not converge: residual={solver.last_residual:.3e}"
        )
    solver.last_x0 = None
    problem.desvars = initial_density.copy()
    problem.iteration = 0

    n_elements = ex * ey * ez
    n_dofs = 3 * (ex + 1) * (ey + 1) * (ez + 1)
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print("TRÄFF ET AL. 2023 CANTILEVER BENCHMARK — PYFANTOM")
    print(f"elements:          {(ex, ey, ez)}")
    print(f"n elements:        {n_elements:,}")
    print(f"elastic DOFs:      {n_dofs:,}")
    print(f"grid levels:       {COARSENINGS + 1}")
    print(f"volume fraction:   {VOLUME_FRACTION:.4f}")
    print(f"filter radius:     {FILTER_RADIUS_ELEMENTS:.2f} elements")
    print(f"load patch elems:  {n_patch_elements:,}")
    print(f"total force:       {total_force:.6f}")
    print(f"CG tolerance:      {CG_TOLERANCE:.1e}")
    print(f"MG cycle:          W (level {W_CYCLE_LEVEL})")
    print(f"MG smoothing:      {SMOOTHING_STEPS} pre + {SMOOTHING_STEPS} post")
    print(
        f"Jacobi damping:    adaptive "
        f"{MINIMUM_JACOBI_DAMPING:.2f}--{JACOBI_DAMPING:.2f}"
    )
    print("coarse solver:     CHOLMOD")
    print("precision:         float64")
    print(f"device:            {device}")
    print(f"setup:             {setup_time:10.3f} s")
    print(f"dummy execution:   {warmup_time:10.3f} s (not timed)")
    print()

    rows = []
    for iteration in range(1, args.max_iterations + 1):
        start = time.perf_counter()
        problem._compute()
        compliance = scalar(problem.f())
        design_density = scalar(cp.mean(problem.get_desvars()))
        physical_density = density_filter.dot(problem.get_desvars())
        physical_mean = scalar(cp.mean(physical_density))
        volume_constraint = physical_mean - VOLUME_FRACTION
        mdn = non_discreteness(physical_density)
        cg_iterations = int(solver.last_iterations)
        residual = float(solver.last_residual)
        if not solver.last_converged:
            synchronize()
            raise RuntimeError(
                f"Physics PCG did not converge at iteration {iteration}: "
                f"iterations={cg_iterations}, residual={residual:.3e}"
            )

        change = oc_update(problem, MOVE_LIMIT)
        synchronize()
        elapsed = time.perf_counter() - start
        row = {
            "iteration": iteration,
            "compliance": compliance,
            "mean_design_density": design_density,
            "mean_physical_density": physical_mean,
            "volume_constraint": volume_constraint,
            "non_discreteness_reported": mdn,
            "change": change,
            "physics_cg_iterations": cg_iterations,
            "physics_relative_residual": residual,
            "filter_cg_iterations": None,
            "filter_relative_residual": None,
            "iteration_time_s": elapsed,
        }
        rows.append(row)
        print(
            f"iter {iteration:4d} | c {compliance:12.6e} | "
            f"rho {design_density:.6f} | phys {physical_mean:.6f} | "
            f"g {volume_constraint:+.3e} | MDN {mdn:8.4f} | "
            f"change {change:.6e} | CG(K) {cg_iterations:4d} | "
            f"relres {residual:.3e} | time {elapsed:.3f} s",
            flush=True,
        )

    write_runtime_csv(args.runtime_csv, rows)


if __name__ == "__main__":
    main()
