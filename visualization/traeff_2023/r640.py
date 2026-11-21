#!/usr/bin/env python3
"""Save every design from a filtered Träff et al. R640 cantilever run."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import topax


jax.config.update("jax_enable_x64", True)


ELEMENTS = np.array((640, 320, 320), dtype=int)
SIZES = np.array((2.0, 1.0, 1.0), dtype=np.float64)
N_LEVELS = 4
GALERKIN_CROSSOVER_LEVEL = 2
N_SMOOTHING_STEPS = 1
SLAB_SIZE = topax.operator.DEFAULT_SLAB_SIZE

ELASTIC_MODULUS = 1.0
POISSON_RATIO = 0.3
RHO_MIN = 1.0e-6
SIMP_PENALTY = 3.0
VOLUME_FRACTION = 0.10
N_DESIGN_ITERATIONS = 50

OC_MOVE = 0.2
OC_STEP = topax.optimizer.oc.construct_oc_step(
    topax.optimizer.oc.init_oc_params(move=OC_MOVE, bisect_tol=1.0e-6)
)

CG_TOLERANCE = 1.0e-5
CG_MAX_ITERATIONS = 1_000
JACOBI_DAMPING = 0.6

# The original 1.5-element filter radius at R64 on a 2 x 1 x 1 domain.
# This is 15 element widths at R640. Topax converts its public Helmholtz
# radius internally to the PDE length r / (2 sqrt(3)).
FILTER_REFERENCE_RESOLUTION = 64
FILTER_REFERENCE_RADIUS_ELEMENTS = 1.5
FILTER_RADIUS_PHYSICAL = float(
    FILTER_REFERENCE_RADIUS_ELEMENTS * SIZES[0] / FILTER_REFERENCE_RESOLUTION
)
FILTER_RADIUS_ELEMENTS = float(FILTER_RADIUS_PHYSICAL / (SIZES[0] / ELEMENTS[0]))
FILTER_PDE_LENGTH = float(FILTER_RADIUS_PHYSICAL / (2.0 * np.sqrt(3.0)))

LOWER_SUPPORT = {
    "center": [0.0, 0.125, 0.5],
    "size": [0.0, 0.250, 1.0],
}
UPPER_SUPPORT = {
    "center": [0.0, 0.875, 0.5],
    "size": [0.0, 0.250, 1.0],
}
DIRICHLET_BOUNDARY_CONDITIONS = (
    (LOWER_SUPPORT, UPPER_SUPPORT),
    (LOWER_SUPPORT, UPPER_SUPPORT),
    (LOWER_SUPPORT, UPPER_SUPPORT),
)

ITERATION_FIELDS = (
    "iteration",
    "evaluated_density_file",
    "updated_density_file",
    "compliance",
    "volume_constraint",
    "mean_design_density",
    "mean_physical_density",
    "non_discreteness",
    "change",
    "physics_cg_iterations",
    "physics_relative_residual",
    "helmholtz_forward_cg_iterations",
    "helmholtz_objective_cg_iterations",
    "helmholtz_constraint_cg_iterations",
    "helmholtz_updated_density_cg_iterations",
    "solve_time_s",
    "save_time_s",
)


def identity_projection(value):
    return value


def prepare_output_directory(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    density_directory = path / "density"
    density_directory.mkdir()
    return path, density_directory


def density_filename(iteration: int) -> str:
    return f"density/iteration_{iteration:03d}.npy"


def save_density(directory: Path, iteration: int, density) -> None:
    destination = directory / f"iteration_{iteration:03d}.npy"
    temporary = directory / f".iteration_{iteration:03d}.{os.getpid()}.npy"
    if destination.exists():
        raise FileExistsError(f"density frame already exists: {destination}")
    host_density = np.asarray(
        jax.device_get(jnp.asarray(density, dtype=jnp.float32))
    )
    with temporary.open("wb") as stream:
        np.save(stream, host_density, allow_pickle=False)
    temporary.replace(destination)


def write_configuration(path: Path, device, max_iterations: int) -> None:
    n_elements = int(np.prod(ELEMENTS))
    n_nodes = int(np.prod(ELEMENTS + 1))
    path.write_text(
        f'''schema_version = 1
problem = "Träff et al. 2023 R640 cantilever with fixed physical Helmholtz filter"
purpose = "visualization"
elements = [{ELEMENTS[0]}, {ELEMENTS[1]}, {ELEMENTS[2]}]
n_elements = {n_elements}
n_displacement_dofs = {3 * n_nodes}
domain_size = [{SIZES[0]}, {SIZES[1]}, {SIZES[2]}]
design_iterations = {max_iterations}
saved_frames = {max_iterations + 1}
density_kind = "filtered physical density"
density_dtype = "float32"
density_axis_order = "x,y,z"
device = "{device}"

[material]
youngs_modulus = {ELASTIC_MODULUS}
poisson_ratio = {POISSON_RATIO}
minimum_relative_stiffness = {RHO_MIN}
simp_penalty = {SIMP_PENALTY}
volume_fraction = {VOLUME_FRACTION}

[filter]
kind = "Helmholtz density filter"
series = "fixed physical radius"
reference_resolution = {FILTER_REFERENCE_RESOLUTION}
reference_radius_elements = {FILTER_REFERENCE_RADIUS_ELEMENTS}
filter_radius_physical = {FILTER_RADIUS_PHYSICAL}
filter_radius_elements = {FILTER_RADIUS_ELEMENTS}
helmholtz_pde_length = {FILTER_PDE_LENGTH}

[multigrid]
levels = {N_LEVELS}
galerkin_crossover_level = {GALERKIN_CROSSOVER_LEVEL}
smoothing_steps = {N_SMOOTHING_STEPS}
jacobi_damping = {JACOBI_DAMPING}
fine_grid_slab_size = {SLAB_SIZE}
cg_tolerance = {CG_TOLERANCE}
cg_max_iterations = {CG_MAX_ITERATIONS}

[optimization]
method = "optimality criteria"
move_limit = {OC_MOVE}
'''
    )


def construct_forces():
    nelx, nely, nelz = map(int, ELEMENTS)
    element_x = np.arange(nelx, dtype=np.float64)[:, None]
    element_y = np.arange(nely, dtype=np.float64)[None, :]
    radius = (nely + 1.0) / 5.0
    distance_squared = (
        (element_x - float(nelx)) ** 2
        + (element_y - (nely - 1.0) / 2.0) ** 2
    )
    patch = distance_squared < radius**2
    n_patch_elements = int(np.count_nonzero(patch))
    if n_patch_elements == 0:
        raise RuntimeError("Träff load patch contains no elements.")

    contribution = -patch.astype(np.float64) / (4.0 * n_patch_elements)
    face_force = np.zeros((nelx + 1, nely + 1), dtype=np.float64)
    face_force[:-1, :-1] += contribution
    face_force[1:, :-1] += contribution
    face_force[:-1, 1:] += contribution
    face_force[1:, 1:] += contribution

    forces = jnp.zeros(
        (3, nelx + 1, nely + 1, nelz + 1),
        dtype=jnp.float64,
    )
    forces = forces.at[2, :, :, 0].set(jnp.asarray(face_force))
    return forces, n_patch_elements


def construct_problem():
    elastic = topax.construct_elastic(ELASTIC_MODULUS, POISSON_RATIO)
    helmholtz = topax.construct_helmholtz(
        FILTER_RADIUS_PHYSICAL,
        thickness=1.0,
    )
    grid = topax.construct_grid(N_LEVELS, SIZES, ELEMENTS)
    local_helmholtz_operators = tuple(
        topax.construct_local_helmholtz(level, helmholtz)
        for level in grid.levels
    )
    fine_local_stiffness = topax.construct_local_stiffness(
        grid.fine_level,
        elastic,
    )
    restrictions = topax.construct_restrictions(
        grid,
        elastic,
        DIRICHLET_BOUNDARY_CONDITIONS,
    )
    forces, n_patch_elements = construct_forces()
    coarse_physics_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )
    coarse_helmholtz_structure = topax.sparse.construct_sparse_helmholtz_structure(
        grid.coarse_level,
        local_helmholtz_operators[-1],
    )
    return (
        elastic,
        helmholtz,
        grid,
        local_helmholtz_operators,
        fine_local_stiffness,
        restrictions,
        forces,
        n_patch_elements,
        coarse_physics_structure,
        coarse_helmholtz_structure,
    )


def construct_optimization_step(
    *,
    helmholtz,
    grid,
    local_helmholtz_operators,
    fine_local_stiffness,
    coarse_physics_structure,
    coarse_helmholtz_structure,
):
    value_and_grad, initial_cg_state = topax.construct_value_and_grad(
        problem=topax.PROBLEM_COMPLIANCE,
        filtering=topax.FILTER_HELMHOLTZ,
        formulation=topax.FORMULATION_MMA,
        constraint_function=topax.volume_constraint,
        helmholtz=helmholtz,
        local_operators_helmholtz=local_helmholtz_operators,
        fine_local_operator=fine_local_stiffness,
        grid=grid,
        coarse_physics_structure=coarse_physics_structure,
        coarse_helmholtz_structure=coarse_helmholtz_structure,
        projection=identity_projection,
        simp_penalty=SIMP_PENALTY,
        rho_min=RHO_MIN,
        tol=CG_TOLERANCE,
        maxiter=CG_MAX_ITERATIONS,
        damping=JACOBI_DAMPING,
        volfrac=VOLUME_FRACTION,
        n_smoothing_steps=N_SMOOTHING_STEPS,
        galerkin_crossover_level=GALERKIN_CROSSOVER_LEVEL,
        slab_size=SLAB_SIZE,
    )

    def optimization_step(
        solvers,
        rho,
        cg_state,
        forces,
        rho_mask,
        restrictions,
    ):
        (
            solvers,
            objective,
            constraint,
            cg_state,
            solve_diagnostics,
        ) = value_and_grad(
            solvers,
            rho,
            forces,
            rho_mask,
            restrictions,
            cg_state,
        )

        rho_new, oc_info = OC_STEP(
            rho,
            objective.grad,
            constraint.value,
            constraint.grad,
        )

        updated_rhs = topax.helmholtz_to_nodal(
            grid.fine_level,
            helmholtz,
            rho_new,
        )
        (
            helmholtz_solver,
            updated_nodal_density,
            updated_filter_info,
        ) = topax.vcycle.helmholtz_vcycle_solve(
            solver=solvers.helmholtz,
            grid=grid,
            local_operators=local_helmholtz_operators,
            coarse_structure=coarse_helmholtz_structure,
            rho_mask=rho_mask,
            rho_nodal=updated_rhs,
            x0=cg_state.helmholtz_value0,
            damping=JACOBI_DAMPING,
            tol=CG_TOLERANCE,
            maxiter=CG_MAX_ITERATIONS,
            n_smoothing_steps=N_SMOOTHING_STEPS,
            slab_size=SLAB_SIZE,
        )
        physical_density_new = topax.helmholtz_to_elemental(
            grid.fine_level,
            updated_nodal_density,
        )
        solvers = solvers._replace(helmholtz=helmholtz_solver)
        cg_state = cg_state.replace(helmholtz_value0=updated_nodal_density)

        physics = solve_diagnostics.physics
        helmholtz_forward = solve_diagnostics.helmholtz_forward
        helmholtz_objective = solve_diagnostics.helmholtz_objective_adjoint
        helmholtz_constraint = solve_diagnostics.helmholtz_constraint_adjoint
        diagnostics = (
            objective.value,
            constraint.value,
            jnp.mean(rho_new),
            jnp.mean(physical_density_new),
            jnp.mean(4.0 * physical_density_new * (1.0 - physical_density_new)),
            oc_info["change"],
            physics["n_iter"],
            physics["converged"],
            physics["relative_residual"],
            helmholtz_forward["n_iter"],
            helmholtz_forward["converged"],
            helmholtz_objective["n_iter"],
            helmholtz_objective["converged"],
            helmholtz_constraint["n_iter"],
            helmholtz_constraint["converged"],
            updated_filter_info["n_iter"],
            updated_filter_info["converged"],
        )
        return (
            solvers,
            rho_new,
            cg_state,
            physical_density_new,
            diagnostics,
        )

    return (
        jax.jit(optimization_step, donate_argnums=(1,)),
        initial_cg_state,
    )


def run(output_directory: Path, max_iterations: int) -> None:
    output_directory, density_directory = prepare_output_directory(output_directory)
    device = jax.devices()[0]
    write_configuration(
        output_directory / "configuration.toml",
        device,
        max_iterations,
    )
    print(f"output:             {output_directory}")
    print(f"device:             {device}")
    print("constructing Träff R640 fixed-radius Helmholtz problem")

    (
        _elastic,
        helmholtz,
        grid,
        local_helmholtz_operators,
        fine_local_stiffness,
        restrictions,
        forces,
        n_patch_elements,
        coarse_physics_structure,
        coarse_helmholtz_structure,
    ) = construct_problem()
    print(f"load patch elements: {n_patch_elements:,}")
    print(f"filter radius:      {FILTER_RADIUS_PHYSICAL:.8f} physical")
    print(f"                    {FILTER_RADIUS_ELEMENTS:.2f} R640 elements")

    rho = jnp.full(tuple(map(int, ELEMENTS)), VOLUME_FRACTION, dtype=jnp.float64)
    rho_mask = jnp.ones_like(rho)
    solvers = topax.solver.setup_cg_filter_solver_state(
        coarse_physics_structure,
        coarse_helmholtz_structure,
        grid,
        restrictions,
    )
    optimization_step, cg_state = construct_optimization_step(
        helmholtz=helmholtz,
        grid=grid,
        local_helmholtz_operators=local_helmholtz_operators,
        fine_local_stiffness=fine_local_stiffness,
        coarse_physics_structure=coarse_physics_structure,
        coarse_helmholtz_structure=coarse_helmholtz_structure,
    )

    print("compiling optimization step")
    compiled_step = optimization_step.lower(
        solvers,
        rho,
        cg_state,
        forces,
        rho_mask,
        restrictions,
    ).compile()

    print("running untimed warm-up")
    warmup_state = jax.tree.map(lambda value: jnp.array(value, copy=True), cg_state)
    warmup_outputs = compiled_step(
        solvers,
        jnp.array(rho, copy=True),
        warmup_state,
        forces,
        rho_mask,
        restrictions,
    )
    jax.block_until_ready(warmup_outputs)
    solvers = warmup_outputs[0]
    warmup_diagnostics = warmup_outputs[-1]
    for name, index in (
        ("physics", 7),
        ("Helmholtz forward", 10),
        ("Helmholtz objective adjoint", 12),
        ("Helmholtz constraint adjoint", 14),
        ("updated-density Helmholtz", 16),
    ):
        if not bool(warmup_diagnostics[index]):
            topax.solver.close_solver_state(solvers)
            raise RuntimeError(f"{name} solve failed during warm-up.")
    del warmup_outputs, warmup_state, warmup_diagnostics

    save_start = time.perf_counter()
    save_density(density_directory, 0, rho)
    print(f"saved iteration 000 in {time.perf_counter() - save_start:.3f} s")

    history_path = output_directory / "iterations.csv"
    completed_iterations = 0
    try:
        with history_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ITERATION_FIELDS)
            writer.writeheader()
            stream.flush()

            for iteration in range(1, max_iterations + 1):
                solve_start = time.perf_counter()
                (
                    solvers,
                    rho,
                    cg_state,
                    physical_density,
                    diagnostics,
                ) = compiled_step(
                    solvers,
                    rho,
                    cg_state,
                    forces,
                    rho_mask,
                    restrictions,
                )
                jax.block_until_ready(
                    (solvers, rho, cg_state, physical_density, diagnostics)
                )
                solve_time = time.perf_counter() - solve_start

                (
                    compliance,
                    volume_constraint,
                    mean_design_density,
                    mean_physical_density,
                    non_discreteness,
                    change,
                    physics_n_iter,
                    physics_converged,
                    physics_residual,
                    forward_n_iter,
                    forward_converged,
                    objective_n_iter,
                    objective_converged,
                    constraint_n_iter,
                    constraint_converged,
                    updated_n_iter,
                    updated_converged,
                ) = diagnostics
                for name, converged in (
                    ("physics", physics_converged),
                    ("Helmholtz forward", forward_converged),
                    ("Helmholtz objective adjoint", objective_converged),
                    ("Helmholtz constraint adjoint", constraint_converged),
                    ("updated-density Helmholtz", updated_converged),
                ):
                    if not bool(converged):
                        raise RuntimeError(
                            f"{name} solve failed at iteration {iteration}."
                        )

                save_start = time.perf_counter()
                save_density(density_directory, iteration, physical_density)
                save_time = time.perf_counter() - save_start
                writer.writerow(
                    {
                        "iteration": iteration,
                        "evaluated_density_file": density_filename(iteration - 1),
                        "updated_density_file": density_filename(iteration),
                        "compliance": float(compliance),
                        "volume_constraint": float(volume_constraint),
                        "mean_design_density": float(mean_design_density),
                        "mean_physical_density": float(mean_physical_density),
                        "non_discreteness": float(non_discreteness),
                        "change": float(change),
                        "physics_cg_iterations": int(physics_n_iter),
                        "physics_relative_residual": float(physics_residual),
                        "helmholtz_forward_cg_iterations": int(forward_n_iter),
                        "helmholtz_objective_cg_iterations": int(objective_n_iter),
                        "helmholtz_constraint_cg_iterations": int(constraint_n_iter),
                        "helmholtz_updated_density_cg_iterations": int(updated_n_iter),
                        "solve_time_s": solve_time,
                        "save_time_s": save_time,
                    }
                )
                stream.flush()
                completed_iterations = iteration
                print(
                    f"iter {iteration:3d} | c {float(compliance):12.6e} | "
                    f"rho {float(mean_physical_density):.6f} | "
                    f"MDN {float(non_discreteness):.6f} | "
                    f"change {float(change):.3e} | "
                    f"CG(K) {int(physics_n_iter):4d} | "
                    f"CG(H) {int(updated_n_iter):3d} | "
                    f"solve {solve_time:.3f} s | save {save_time:.3f} s"
                )
    finally:
        topax.solver.close_solver_state(solvers)

    if completed_iterations == max_iterations:
        (output_directory / "COMPLETE").write_text(
            f"completed_iterations = {completed_iterations}\n"
        )
    print(f"completed {completed_iterations} design iterations")
    print(f"density history:    {density_directory}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="exact output directory for this run (must not already exist)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=N_DESIGN_ITERATIONS,
        help=f"number of saved design updates (default: {N_DESIGN_ITERATIONS})",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.max_iterations <= N_DESIGN_ITERATIONS:
        parser.error(
            f"--max-iterations must be between 1 and {N_DESIGN_ITERATIONS}"
        )
    return args


def main():
    args = parse_args()
    run(args.output_dir, args.max_iterations)


if __name__ == "__main__":
    main()
