import einops
import jax
import jax.numpy as jnp
import numpy as np

from topax.grid import Grid
from topax.material.helmholtz import (
    construct_helmholtz,
    construct_local_helmholtz,
    helmholtz_to_elemental,
    helmholtz_to_nodal,
)
from topax.operator import matrix_free_matvec


jax.config.update("jax_enable_x64", True)


def test_construct_helmholtz_hex8_linear_field_gradient_energy():
    sizes = np.array([2.0, 0.5, 1.5])
    grid = Grid(n_levels=1, sizes=sizes, elements=np.ones(3, dtype=int))
    helmholtz = construct_helmholtz(filter_radius=0.6, thickness=7.0)
    mass = construct_helmholtz(filter_radius=0.0, thickness=7.0)

    local = construct_local_helmholtz(grid.fine_level, helmholtz)
    local_mass = construct_local_helmholtz(grid.fine_level, mass)
    assert local.shape == (8, 1, 2, 2, 2)

    gradient = einops.rearrange(
        local - local_mass, "oc ic x y z -> oc (ic x y z)"
    )
    r = helmholtz.filter_radius / (2.0 * np.sqrt(3.0))
    volume = np.prod(sizes)

    for axis, length in enumerate(sizes):
        field = jnp.array(
            [indices[axis] for indices in np.ndindex(2, 2, 2)],
            dtype=jnp.float64,
        )
        np.testing.assert_allclose(
            field @ gradient @ field,
            r**2 * volume / length**2,
            rtol=1e-12,
            atol=1e-12,
        )


def test_helmholtz_to_elemental_uses_eight_hex8_nodes():
    grid = Grid(
        n_levels=1,
        sizes=np.ones(3),
        elements=np.array([2, 1, 1]),
    )
    nodal = jnp.arange(12, dtype=jnp.float64).reshape(1, 3, 2, 2)

    actual = helmholtz_to_elemental(grid.fine_level, nodal)

    expected = np.array([[[3.5]], [[7.5]]])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_helmholtz_3d_preserves_constant_density_on_odd_grid():
    grid = Grid(
        n_levels=1,
        sizes=np.array([3.0, 1.0, 2.0]),
        elements=np.array([3, 2, 1]),
    )
    level = grid.fine_level
    helmholtz = construct_helmholtz(filter_radius=0.7, thickness=5.0)
    local = construct_local_helmholtz(level, helmholtz)
    rho = jnp.full(tuple(level.elements), 0.37, dtype=jnp.float64)
    rho_mask = jnp.ones_like(rho)
    nodal_shape = tuple(level.nodes)
    basis = jnp.eye(level.n_nodes, dtype=jnp.float64)

    def matvec_flat(x):
        return matrix_free_matvec(
            level,
            x.reshape((1, *nodal_shape)),
            local,
            rho_mask,
        ).reshape(-1)

    matrix = jnp.stack(
        [matvec_flat(basis[:, i]) for i in range(level.n_nodes)], axis=1
    )
    rhs = helmholtz_to_nodal(level, helmholtz, rho)
    filtered_nodal = jnp.linalg.solve(matrix, rhs.reshape(-1)).reshape(
        (1, *nodal_shape)
    )

    np.testing.assert_allclose(filtered_nodal, 0.37, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        helmholtz_to_elemental(level, filtered_nodal),
        rho,
        rtol=1e-11,
        atol=1e-11,
    )
