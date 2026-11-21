import jax
import jax.numpy as jnp
import numpy as np
import pytest

import topax


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "galerkin_crossover_level",
    [None, 1, 2],
)
def test_complete_galerkin_mgpcg_matches_direct_heterogeneous_solve(
    galerkin_crossover_level,
):
    elements = np.array([4, 4, 4])
    sizes = np.array([2.0, 1.5, 1.0])
    grid = topax.construct_grid(
        n_levels=3,
        sizes=sizes,
        elements=elements,
    )
    elastic = topax.construct_elastic(
        elastic_module=2.5,
        poisson_ratio=0.28,
    )
    local_stiffnesses = tuple(
        topax.construct_local_stiffness(level, elastic) for level in grid.levels
    )
    fixed_face = {
        "center": [0.0, 0.5, 0.5],
        "size": [0.0, 1.0, 1.0],
    }
    restrictions = topax.construct_restrictions(
        grid,
        elastic,
        [[fixed_face], [fixed_face], [fixed_face]],
    )
    rho = jnp.linspace(
        0.2,
        0.95,
        grid.fine_level.n_elements,
        dtype=jnp.float64,
    ).reshape(tuple(elements))
    right_hand_side = jnp.cos(
        jnp.arange(
            3 * grid.fine_level.n_nodes,
            dtype=jnp.float64,
        )
        / 17.0
    ).reshape(3, *map(int, grid.fine_level.nodes))
    right_hand_side = right_hand_side * restrictions[0]

    fine_structure = topax.sparse.construct_sparse_physics_structure(
        grid.fine_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[0],
    )
    coarse_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )
    direct_solver = topax.solver.setup_physics_solver(fine_structure)
    galerkin_solver = topax.solver.setup_galerkin_physics_solver(
        coarse_structure, grid, restrictions
    )

    @jax.jit
    def direct_solve(solver, rho, rhs):
        weighing = topax.operator.physics_weighing(
            rho.reshape(-1),
            simp_penalty=3.0,
            rho_min=1e-4,
        )
        solver = topax.solver.refactor_physics_solver(
            solver,
            fine_structure,
            local_stiffnesses[0],
            weighing,
        )
        return topax.solver.solve_sparse(solver, fine_structure, rhs)

    @jax.jit
    def galerkin_solve(solver, rho, rhs):
        return topax.vcycle.physics_vcycle_solve(
            solver=solver,
            grid=grid,
            fine_local_operator=local_stiffnesses[0],
            coarse_structure=coarse_structure,
            rho=rho,
            forces=rhs,
            restrictions=restrictions,
            x0=jnp.zeros_like(rhs),
            simp_penalty=3.0,
            rho_min=1e-4,
            damping=0.8,
            tol=1e-10,
            maxiter=1_000,
            n_smoothing_steps=2,
            galerkin_crossover_level=galerkin_crossover_level,
        )

    try:
        direct_solver, expected = direct_solve(
            direct_solver,
            rho,
            right_hand_side,
        )
        galerkin_solver, actual, info = galerkin_solve(
            galerkin_solver,
            rho,
            right_hand_side,
        )
        actual, expected = jax.block_until_ready((actual, expected))

        assert bool(info["converged"])
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)
    finally:
        direct_solver.close()
        galerkin_solver.close()


def test_physics_galerkin_hierarchy_rejects_non_dyadic_grid():
    grid = topax.construct_grid(
        n_levels=2,
        sizes=np.ones(3),
        elements=np.array([3, 2, 2]),
    )
    elastic = topax.construct_elastic(
        elastic_module=1.0,
        poisson_ratio=0.3,
    )
    local_stiffnesses = tuple(
        topax.construct_local_stiffness(level, elastic) for level in grid.levels
    )

    with pytest.raises(ValueError, match="fully dyadic"):
        topax.vcycle.construct_physics_multigrid_data(
            solver=None,
            grid=grid,
            fine_local_operator=local_stiffnesses[0],
            coarse_structure=None,
            rho=jnp.ones((3, 2, 2)),
            simp_penalty=3.0,
            rho_min=1e-4,
        )

@pytest.mark.parametrize(
    "galerkin_crossover_level",
    [0, 3, -1, True, 1.5, jnp.asarray(1)],
)
def test_physics_galerkin_hierarchy_rejects_invalid_crossover(
    galerkin_crossover_level,
):
    grid = topax.construct_grid(
        n_levels=3,
        sizes=np.ones(3),
        elements=np.array([4, 4, 4]),
    )
    elastic = topax.construct_elastic(1.0, 0.3)
    local_stiffnesses = tuple(
        topax.construct_local_stiffness(level, elastic) for level in grid.levels
    )

    with pytest.raises(ValueError, match="galerkin_crossover_level"):
        topax.vcycle.construct_physics_multigrid_data(
            solver=None,
            grid=grid,
            fine_local_operator=local_stiffnesses[0],
            coarse_structure=None,
            rho=jnp.ones((4, 4, 4)),
            simp_penalty=3.0,
            rho_min=1e-4,
            galerkin_crossover_level=galerkin_crossover_level,
        )
