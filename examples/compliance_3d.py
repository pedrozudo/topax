"""Small 3D compliance topology optimization using MMA."""

from __future__ import annotations

import argparse
import numbers

import jax
import jax.numpy as jnp
import numpy as np

import topax


DEFAULT_ELEMENTS = (8, 4, 4)
DEFAULT_LEVELS = 2
DEFAULT_ITERATIONS = 20
DEFAULT_VOLUME_FRACTION = 0.5


def _validate_configuration(elements, n_levels, iterations, volume_fraction):
    elements = np.asarray(elements)
    if (
        elements.shape != (3,)
        or not np.issubdtype(elements.dtype, np.integer)
        or np.any(elements <= 0)
    ):
        raise ValueError("elements must contain three positive integers.")

    if isinstance(iterations, bool) or not isinstance(iterations, numbers.Integral):
        raise ValueError("iterations must be a positive integer.")
    if iterations < 1:
        raise ValueError("iterations must be a positive integer.")

    if not 0.0 < volume_fraction < 1.0:
        raise ValueError("volume_fraction must lie strictly between zero and one.")

    return (
        elements.astype(int),
        int(n_levels),
        int(iterations),
        float(volume_fraction),
    )


def run(
    elements=DEFAULT_ELEMENTS,
    n_levels=DEFAULT_LEVELS,
    iterations=DEFAULT_ITERATIONS,
    volume_fraction=DEFAULT_VOLUME_FRACTION,
    *,
    verbose=True,
):
    """Run a small compliance-minimization problem and return the final design."""

    (
        elements,
        n_levels,
        iterations,
        volume_fraction,
    ) = _validate_configuration(
        elements,
        n_levels,
        iterations,
        volume_fraction,
    )

    sizes = np.ones(3)
    elastic = topax.construct_elastic(
        elastic_module=1.0,
        poisson_ratio=0.3,
        thickness=1.0,
    )
    grid = topax.construct_grid(
        n_levels=n_levels,
        sizes=sizes,
        elements=elements,
    )
    fine_local_stiffness = topax.construct_local_stiffness(
        grid.fine_level,
        elastic,
    )

    fixed_face = {"center": [0.0, 0.5, 0.5], "size": [0.0, 1.0, 1.0]}
    restrictions = topax.construct_restrictions(
        grid,
        elastic,
        [[fixed_face], [fixed_face], [fixed_face]],
    )
    forces = topax.construct_forces(
        grid,
        elastic,
        [
            [],
            [],
            [
                {
                    "center": [1.0, 0.5, 0.0],
                    "size": [0.0, 0.0, 0.0],
                    "intensity": -1.0,
                }
            ],
        ],
    )
    coarse_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )
    solvers = topax.solver.setup_physics_solver_state(
        coarse_structure,
        grid,
        restrictions,
    )

    value_and_grad, cg_state = topax.construct_value_and_grad(
        problem=topax.PROBLEM_COMPLIANCE,
        filtering=topax.FILTER_NONE,
        formulation=topax.FORMULATION_MMA,
        fine_local_operator=fine_local_stiffness,
        grid_physics=grid,
        coarse_physics_structure=coarse_structure,
        simp_penalty=3.0,
        rho_min=1e-6,
        # The default JAX mode is float32; request a tolerance it can verify.
        tol=1e-5,
        maxiter=10_000,
        damping=0.8,
        volfrac=volume_fraction,
    )
    value_and_grad = jax.jit(value_and_grad)

    rho = jnp.full(tuple(elements), volume_fraction)
    rho_mask = jnp.ones_like(rho, dtype=jnp.bool_)

    mma_parameters = topax.optimizer.mma.init_mma_1con_params(move=0.2)
    mma_state = topax.optimizer.mma.init_mma_1con_state(
        rho,
        mma_parameters,
    )
    mma_step_impl = topax.optimizer.mma.construct_mma_step(mma_parameters)

    def mma_update(rho, objective, constraint, mma_state):
        rho_new, mma_state, info = mma_step_impl(
            rho,
            objective.value,
            objective.grad,
            constraint.value,
            constraint.grad,
            mma_state,
        )
        change = jnp.max(jnp.abs(rho_new - rho))
        return rho_new, mma_state, info, change

    mma_update = jax.jit(mma_update)

    history = []

    if verbose:
        print(" iter |   compliance | mean density | constraint | max change")

    try:
        for iteration in range(1, iterations + 1):
            (
                solvers,
                objective,
                constraint,
                cg_state,
                linear_solve_diagnostics,
            ) = value_and_grad(
                solvers,
                rho,
                forces,
                rho_mask,
                restrictions,
                cg_state,
            )
            (
                objective,
                constraint,
                linear_solves_converged,
                mean_density,
            ) = jax.block_until_ready(
                (
                    objective,
                    constraint,
                    linear_solve_diagnostics.converged,
                    jnp.mean(rho),
                )
            )
            if not bool(linear_solves_converged):
                raise RuntimeError(
                    f"linear solve did not converge at iteration {iteration}"
                )

            rho_new, mma_state, mma_info, change = mma_update(
                rho,
                objective,
                constraint,
                mma_state,
            )
            rho_new, change, mma_feasible = jax.block_until_ready(
                (rho_new, change, mma_info["feasible"])
            )
            if not bool(mma_feasible):
                raise RuntimeError(
                    f"MMA subproblem was infeasible at iteration {iteration}"
                )

            entry = {
                "iteration": iteration,
                "compliance": float(objective.value),
                "mean_density": float(mean_density),
                "volume_constraint": float(constraint.value),
                "change": float(change),
            }
            history.append(entry)

            if verbose:
                print(
                    f"{iteration:5d} | "
                    f"{entry['compliance']:12.5e} | "
                    f"{entry['mean_density']:12.6f} | "
                    f"{entry['volume_constraint']:+10.3e} | "
                    f"{entry['change']:10.3e}"
                )

            rho = rho_new

        (
            solvers,
            objective,
            constraint,
            cg_state,
            linear_solve_diagnostics,
        ) = value_and_grad(
            solvers,
            rho,
            forces,
            rho_mask,
            restrictions,
            cg_state,
        )
        objective, constraint, linear_solves_converged, rho = jax.block_until_ready(
            (
                objective,
                constraint,
                linear_solve_diagnostics.converged,
                rho,
            )
        )
        if not bool(linear_solves_converged):
            raise RuntimeError("linear solve did not converge for the final design")

        return {
            "density": rho,
            "history": tuple(history),
            "compliance": objective.value,
            "volume_constraint": constraint.value,
            "compliance_gradient": objective.grad,
            "constraint_gradient": constraint.grad,
        }
    finally:
        topax.solver.close_solver_state(solvers)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--elements",
        type=int,
        nargs=3,
        default=DEFAULT_ELEMENTS,
        metavar=("EX", "EY", "EZ"),
        help="number of elements along each axis",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=DEFAULT_LEVELS,
        help="total number of multigrid levels, including the fine grid",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="number of MMA iterations",
    )
    parser.add_argument(
        "--volume-fraction",
        type=float,
        default=DEFAULT_VOLUME_FRACTION,
        help="target material volume fraction",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run(
        elements=args.elements,
        n_levels=args.levels,
        iterations=args.iterations,
        volume_fraction=args.volume_fraction,
    )
    print()
    print("final compliance:", float(result["compliance"]))
    print("final volume constraint:", float(result["volume_constraint"]))


if __name__ == "__main__":
    main()
