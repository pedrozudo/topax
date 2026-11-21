
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from topax.grid import Grid
from topax.vcycle import (
    prolongate_correction,
    prolongation_kernel,
    restrict_residual,
)

jax.config.update("jax_enable_x64", True)


def construct_cycle(elements):
    return Grid(
        n_levels=2,
        sizes=np.ones(3),
        elements=np.asarray(elements),
    )


def test_vcycle_3d_uses_tensor_product_interpolation_kernel():
    grid = construct_cycle([4, 4, 4])
    weights = jnp.array([1.0, 2.0, 1.0])
    expected = jnp.einsum("i,j,k->ijk", weights, weights, weights) / 8.0

    np.testing.assert_allclose(
        prolongation_kernel(grid.n_dims, weights.dtype)[0, 0],
        expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("elements", [[4, 4, 4], [3, 5, 1]])
def test_vcycle_3d_prolongation_preserves_constants(elements):
    grid = construct_cycle(elements)
    coarse = jnp.ones((1, *grid.levels[1].nodes))

    fine = prolongate_correction(coarse, grid, None, 0)

    assert fine.shape == (1, *grid.fine_level.nodes)
    np.testing.assert_allclose(fine, 1.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("elements", [[4, 4, 4], [3, 5, 1]])
def test_vcycle_3d_restriction_is_prolongation_transpose(elements):
    grid = construct_cycle(elements)
    fine = jnp.arange(grid.fine_level.n_nodes, dtype=jnp.float64).reshape(
        (1, *grid.fine_level.nodes)
    )
    coarse = jnp.arange(grid.levels[1].n_nodes, dtype=jnp.float64).reshape(
        (1, *grid.levels[1].nodes)
    )

    lhs = jnp.vdot(prolongate_correction(coarse, grid, None, 0), fine)
    rhs = jnp.vdot(coarse, restrict_residual(fine, grid, None, 0))

    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)
