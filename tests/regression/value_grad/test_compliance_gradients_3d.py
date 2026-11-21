import jax
import jax.numpy as jnp
import numpy as np
import pytest

import topax


jax.config.update("jax_enable_x64", True)


ELEMENTS = np.array([2, 2, 2])
SIZES = np.array([1.0, 1.0, 1.0])
FD_STEP = 1e-4


def construct_problem(filtering, *, tol=1e-11, maxiter=10_000):
    elastic = topax.construct_elastic(
        elastic_module=1.0,
        poisson_ratio=0.3,
        thickness=1.0,
    )
    grid = topax.construct_grid(
        n_levels=2,
        sizes=SIZES,
        elements=ELEMENTS,
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
            [{"center": [1.0, 0.5, 0.0], "size": [0.0, 0.0, 0.0], "intensity": -1.0}],
        ],
    )
    coarse_physics_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )

    kwargs = dict(
        problem=topax.PROBLEM_COMPLIANCE,
        filtering=filtering,
        formulation=topax.FORMULATION_MMA,
        fine_local_operator=fine_local_stiffness,
        coarse_physics_structure=coarse_physics_structure,
        simp_penalty=3,
        rho_min=1e-6,
        tol=tol,
        maxiter=maxiter,
        damping=0.8,
        volfrac=0.5,
    )
    if filtering == topax.FILTER_NONE:
        kwargs["grid_physics"] = grid
        solvers = topax.solver.setup_physics_solver_state(
            coarse_physics_structure, grid, restrictions
        )
    elif filtering == topax.FILTER_CONVOLUTION:
        kwargs.update(
            grid_physics=grid,
            conv_filter=topax.construct_conv_filter(SIZES, ELEMENTS, radius=0.8),
            projection=lambda rho: rho,
        )
        solvers = topax.solver.setup_physics_solver_state(
            coarse_physics_structure, grid, restrictions
        )
    elif filtering == topax.FILTER_HELMHOLTZ:
        helmholtz = topax.construct_helmholtz(filter_radius=0.8, thickness=1.0)
        local_helmholtz = tuple(
            topax.construct_local_helmholtz(level, helmholtz) for level in grid.levels
        )
        coarse_helmholtz_structure = topax.sparse.construct_sparse_helmholtz_structure(
            grid.coarse_level,
            local_helmholtz[-1],
        )
        kwargs.update(
            grid=grid,
            helmholtz=helmholtz,
            local_operators_helmholtz=local_helmholtz,
            coarse_helmholtz_structure=coarse_helmholtz_structure,
            projection=lambda rho: rho,
        )
        solvers = topax.solver.setup_cg_filter_solver_state(
            coarse_physics_structure,
            coarse_helmholtz_structure,
            grid,
            restrictions,
        )
    else:
        raise AssertionError(f"Unhandled filter {filtering}")

    value_and_grad, initial_state = topax.construct_value_and_grad(**kwargs)
    return (
        jax.jit(value_and_grad),
        forces,
        jnp.ones(tuple(ELEMENTS), dtype=jnp.float64),
        restrictions,
        solvers,
        initial_state,
    )


@pytest.mark.parametrize(
    "filtering",
    [
        pytest.param(topax.FILTER_NONE, id="no-filter"),
        pytest.param(topax.FILTER_CONVOLUTION, id="convolution"),
        pytest.param(topax.FILTER_HELMHOLTZ, id="helmholtz"),
    ],
)
def test_compliance_3d_gradients_match_finite_differences(filtering):
    value_and_grad, forces, rho_mask, restrictions, solvers, initial_state = (
        construct_problem(filtering)
    )
    rho = jnp.linspace(0.35, 0.75, np.prod(ELEMENTS)).reshape(tuple(ELEMENTS))
    direction = jnp.array(
        [0.4, -0.2, 0.7, -0.5, 0.1, 0.6, -0.3, 0.8],
        dtype=jnp.float64,
    ).reshape(tuple(ELEMENTS))
    direction = direction / jnp.linalg.norm(direction)

    state = initial_state
    try:
        solvers, objective, constraint, state, diagnostics = value_and_grad(
            solvers, rho, forces, rho_mask, restrictions, state
        )
        assert bool(diagnostics.converged)

        def evaluate(rho_perturbed, solvers, state):
            (
                solvers,
                objective_perturbed,
                constraint_perturbed,
                state,
                _,
            ) = value_and_grad(
                solvers,
                rho_perturbed,
                forces,
                rho_mask,
                restrictions,
                state,
            )
            return (
                solvers,
                objective_perturbed.value,
                constraint_perturbed.value,
                state,
            )

        solvers, plus_objective, plus_constraint, state = evaluate(
            rho + FD_STEP * direction,
            solvers,
            state,
        )
        solvers, minus_objective, minus_constraint, state = evaluate(
            rho - FD_STEP * direction,
            solvers,
            state,
        )
        objective_fd = (plus_objective - minus_objective) / (2.0 * FD_STEP)
        constraint_fd = (plus_constraint - minus_constraint) / (2.0 * FD_STEP)

        np.testing.assert_allclose(
            jnp.vdot(objective.grad, direction),
            objective_fd,
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            jnp.vdot(constraint.grad, direction),
            constraint_fd,
            rtol=2e-7,
            atol=2e-9,
        )
    finally:
        topax.solver.close_solver_state(solvers)


@pytest.mark.parametrize(
    "filtering",
    [
        pytest.param(topax.FILTER_NONE, id="no-filter"),
        pytest.param(topax.FILTER_CONVOLUTION, id="convolution"),
        pytest.param(topax.FILTER_HELMHOLTZ, id="helmholtz"),
    ],
)
def test_compliance_diagnostics_report_non_convergence(filtering):
    value_and_grad, forces, rho_mask, restrictions, solvers, state = (
        construct_problem(filtering, maxiter=0)
    )
    rho = jnp.full(tuple(ELEMENTS), 0.5, dtype=jnp.float64)

    try:
        solvers, _, _, _, diagnostics = value_and_grad(
            solvers,
            rho,
            forces,
            rho_mask,
            restrictions,
            state,
        )
        assert not bool(diagnostics.converged)
        assert int(diagnostics.physics["n_iter"]) == 0

        helmholtz_infos = (
            diagnostics.helmholtz_forward,
            diagnostics.helmholtz_objective_adjoint,
            diagnostics.helmholtz_constraint_adjoint,
        )
        if filtering == topax.FILTER_HELMHOLTZ:
            for info in helmholtz_infos:
                assert info is not None
                assert int(info["n_iter"]) == 0
        else:
            assert helmholtz_infos == (None, None, None)
    finally:
        topax.solver.close_solver_state(solvers)
