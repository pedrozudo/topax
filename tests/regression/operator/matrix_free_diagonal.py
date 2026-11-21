import numpy as np

import jax
import jax.numpy as jnp

from topax.grid import Grid
from topax.material.elastic import (
    construct_elastic,
    construct_local_stiffness,
    elastic_n_dofs_node,
)
from topax.material.helmholtz import (
    construct_helmholtz,
    construct_local_helmholtz,
    helmholtz_n_dofs_node,
)
from topax.operator import (
    matrix_free_diagonal,
    matrix_free_matvec,
    physics_weighing,
)


jax.config.update(
    "jax_enable_x64",
    True,
)


def _materialize_operator(
    operator,
    nodal_shape,
):
    """
    Materialize a small matrix-free operator into a dense matrix.

    This is only intended for regression tests on tiny problems.
    """
    n_dofs = int(np.prod(nodal_shape))

    basis = jnp.eye(
        n_dofs,
        dtype=jnp.float64,
    )

    def matvec_flat(x):
        x = x.reshape(nodal_shape)

        y = operator(x)

        return y.reshape(-1)

    return jnp.stack(
        [matvec_flat(basis[:, i]) for i in range(n_dofs)],
        axis=1,
    )


def _explicit_diagonal(
    operator,
    nodal_shape,
):
    """
    Compute the operator diagonal by explicitly materializing
    the corresponding dense matrix.
    """
    matrix = _materialize_operator(
        operator,
        nodal_shape,
    )

    return jnp.diag(matrix).reshape(nodal_shape)


def test_physics_matrix_free_diagonal_rectangular_matches_operator():
    """
    The assembled diagonal must match the diagonal of the full
    matrix-free elasticity operator for rectangular elements.
    """
    sizes = jnp.array(
        [
            4.0,
            1.5,
        ],
        dtype=jnp.float64,
    )

    elements = np.array(
        [
            2,
            3,
        ],
        dtype=int,
    )

    # hx = 2.0
    # hy = 0.5
    elastic = construct_elastic(
        elastic_module=1.0,
        poisson_ratio=0.3,
        thickness=1.0,
    )

    grid = Grid(
        n_levels=1,
        sizes=sizes,
        elements=elements,
    )

    level = grid.fine_level

    local_stiffness = construct_local_stiffness(
        level,
        elastic,
    )

    # A nonuniform density field verifies that local diagonal
    # contributions are assembled onto the correct global nodes.
    rho = jnp.array(
        [
            [
                0.20,
                0.50,
                0.80,
            ],
            [
                0.35,
                0.65,
                1.00,
            ],
        ],
        dtype=jnp.float64,
    )

    weighing = physics_weighing(rho, 3.0, 1e-3)
    actual = matrix_free_diagonal(level, local_stiffness, weighing)

    nodal_shape = (
        elastic_n_dofs_node(level.n_dims),
        *level.nodes,
    )

    expected = _explicit_diagonal(
        lambda x: matrix_free_matvec(level, x, local_stiffness, weighing),
        nodal_shape,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    # Rectangular elements generally give different diagonal
    # contributions for the x- and y-displacement components.
    assert not np.allclose(
        np.asarray(actual[0]),
        np.asarray(actual[1]),
    )


def test_physics_matrix_free_diagonal_square_matches_operator():
    """
    The assembled diagonal must match the diagonal of the full
    matrix-free elasticity operator for square elements.
    """
    sizes = jnp.array(
        [
            3.0,
            2.0,
        ],
        dtype=jnp.float64,
    )

    elements = np.array(
        [
            3,
            2,
        ],
        dtype=int,
    )

    # hx = hy = 1.0
    elastic = construct_elastic(
        elastic_module=7.5,
        poisson_ratio=0.27,
        thickness=1.3,
    )

    grid = Grid(
        n_levels=1,
        sizes=sizes,
        elements=elements,
    )

    level = grid.fine_level

    local_stiffness = construct_local_stiffness(
        level,
        elastic,
    )

    rho = jnp.array(
        [
            [
                0.25,
                0.50,
            ],
            [
                0.40,
                0.75,
            ],
            [
                0.90,
                1.00,
            ],
        ],
        dtype=jnp.float64,
    )

    weighing = physics_weighing(rho, 3.0, 1e-3)
    actual = matrix_free_diagonal(level, local_stiffness, weighing)

    nodal_shape = (
        elastic_n_dofs_node(level.n_dims),
        *level.nodes,
    )

    expected = _explicit_diagonal(
        lambda x: matrix_free_matvec(level, x, local_stiffness, weighing),
        nodal_shape,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    # For square isotropic Q4 elements, the two displacement
    # components are symmetric.
    np.testing.assert_allclose(
        actual[0],
        actual[1],
        rtol=1e-12,
        atol=1e-12,
    )


def test_helmholtz_matrix_free_diagonal_rectangular_matches_operator():
    """
    The assembled diagonal must match the diagonal of the full
    matrix-free Helmholtz operator for rectangular elements.
    """
    sizes = jnp.array(
        [
            4.0,
            1.5,
        ],
        dtype=jnp.float64,
    )

    elements = np.array(
        [
            2,
            3,
        ],
        dtype=int,
    )

    helmholtz = construct_helmholtz(
        filter_radius=0.8,
        thickness=1.7,
    )

    grid = Grid(
        n_levels=1,
        sizes=sizes,
        elements=elements,
    )

    level = grid.fine_level

    local_helmholtz = construct_local_helmholtz(
        level,
        helmholtz,
    )

    rho_mask = jnp.array(
        [
            [
                1.0,
                0.7,
                1.0,
            ],
            [
                0.4,
                1.0,
                0.8,
            ],
        ],
        dtype=jnp.float64,
    )

    actual = matrix_free_diagonal(level, local_helmholtz, rho_mask)

    nodal_shape = (
        helmholtz_n_dofs_node(level.n_dims),
        *level.nodes,
    )

    expected = _explicit_diagonal(
        lambda x: matrix_free_matvec(level, x, local_helmholtz, rho_mask),
        nodal_shape,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
