import jax
import jax.numpy as jnp
import numpy as np
import pytest

import topax.solver as solver_api
import topax.sparse as sparse
from topax.boundary import construct_restrictions
from topax.grid import Grid
from topax.material.elastic import construct_elastic, construct_local_stiffness
from topax.operator import matrix_free_diagonal, matrix_free_matvec, simp


jax.config.update(
    "jax_enable_x64",
    True,
)


def _construct_fixed_x_face(grid, elastic):
    fixed_face = {
        "center": [0.0, 0.5, 0.5],
        "size": [0.0, 1.0, 1.0],
    }

    return construct_restrictions(
        grid,
        elastic,
        [
            [fixed_face],
            [fixed_face],
            [fixed_face],
        ],
    )[0]


def _construct_operators(elements):
    elements = np.asarray(elements, dtype=int)

    grid = Grid(
        n_levels=1,
        sizes=np.array([2.6, 1.4, 3.2]),
        elements=elements,
    )

    elastic = construct_elastic(
        elastic_module=7.5,
        poisson_ratio=0.27,
        thickness=1.0,
    )

    local_stiffness = construct_local_stiffness(
        grid.fine_level,
        elastic,
    )

    restriction = _construct_fixed_x_face(
        grid,
        elastic,
    )

    rho = jnp.linspace(
        0.2,
        0.9,
        grid.fine_level.n_elements,
        dtype=jnp.float64,
    ).reshape(tuple(elements))

    sparse_structure = sparse.construct_sparse_physics_structure(
        grid.fine_level,
        n_dofs_node=grid.n_dims,
        restriction=restriction,
    )

    weighing = simp(
        rho.reshape(-1),
        simp_penalty=3.0,
        rho_min=1e-3,
    )
    sparse_values = sparse.compute_physics_csr_values(
        sparse_structure,
        local_stiffness,
        weighing,
    )

    return (
        grid,
        restriction,
        sparse_structure,
        sparse_values,
        local_stiffness,
        weighing.reshape(tuple(elements)),
    )


@pytest.mark.parametrize(
    "elements",
    [
        (1, 1, 1),
        (2, 1, 2),
        (2, 2, 1),
    ],
)
def test_sparse_and_matrix_free_3d_matvecs_match(elements):
    (
        grid,
        restriction,
        sparse_structure,
        sparse_values,
        local_stiffness,
        weighing,
    ) = _construct_operators(
        elements,
    )

    displacement = jnp.sin(
        jnp.arange(
            3 * grid.fine_level.n_nodes,
            dtype=jnp.float64,
        )
        / 7.0
    ).reshape(
        3,
        *grid.fine_level.nodes,
    )

    displacement = displacement * restriction

    sparse_matrix = sparse.sparse_todense(sparse_structure, sparse_values)
    sparse_result = sparse_matrix @ displacement.reshape(-1)
    matrix_free_result = matrix_free_matvec(
        grid.fine_level,
        displacement,
        local_stiffness,
        weighing,
        restriction,
    ).reshape(-1)

    np.testing.assert_allclose(
        matrix_free_result,
        sparse_result,
        rtol=1e-12,
        atol=1e-12,
    )

    sparse_diagonal = jnp.diag(sparse_matrix).reshape(
        3,
        *grid.fine_level.nodes,
    )

    np.testing.assert_allclose(
        matrix_free_diagonal(
            grid.fine_level,
            local_stiffness,
            weighing,
        )[restriction],
        sparse_diagonal[restriction],
        rtol=1e-12,
        atol=1e-12,
    )


def test_xolky_3d_solve_satisfies_matrix_free_system():
    (
        grid,
        restriction,
        sparse_structure,
        sparse_values,
        local_stiffness,
        weighing,
    ) = _construct_operators(
        elements=(2, 2, 2),
    )

    rhs = jnp.cos(
        jnp.arange(
            3 * grid.fine_level.n_nodes,
            dtype=jnp.float64,
        )
        / 11.0
    ).reshape(
        3,
        *grid.fine_level.nodes,
    )

    rhs = rhs * restriction

    solver = solver_api.setup_physics_solver(sparse_structure)
    try:
        solver = solver_api.refactor_sparse_solver(solver, sparse_values)

        @jax.jit
        def solve(solver, structure, right_hand_side):
            return solver_api.solve_sparse(
                solver,
                structure,
                right_hand_side,
            )

        solver, solution = solve(
            solver,
            sparse_structure,
            rhs,
        )

        matrix_free_result = matrix_free_matvec(
            grid.fine_level,
            solution,
            local_stiffness,
            weighing,
            restriction,
        )

        np.testing.assert_allclose(
            matrix_free_result,
            rhs,
            rtol=1e-10,
            atol=1e-10,
        )
    finally:
        solver.close()

@pytest.mark.parametrize("elements", [(1, 1, 1), (2, 2, 1)])
def test_heterogeneous_physics_csr_values_match_scalar_weighing(elements):
    (
        _,
        _,
        sparse_structure,
        expected,
        local_stiffness,
        weighing,
    ) = _construct_operators(elements)
    element_matrices = (
        weighing.reshape(-1, 1, 1)
        * local_stiffness.reshape(1, 24, 24)
    )

    actual = sparse.compute_heterogeneous_physics_csr_values(
        sparse_structure,
        element_matrices,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
