import jax
import jax.numpy as jnp
import numpy as np
import pytest

from topax.vcycle import contract_densities


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (4, 6),
        (4, 4, 4),
        (6, 4, 2),
    ],
)
def test_contract_densities_even_grid_preserves_constant(shape):
    rho = jnp.full(shape, 0.375, dtype=jnp.float64)

    coarse = contract_densities(rho, len(shape))

    np.testing.assert_allclose(
        coarse,
        0.375,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (3, 5),
        (3, 5, 1),
        (5, 3, 3),
    ],
)
def test_contract_densities_odd_grid_preserves_constant(shape):
    rho = jnp.full(shape, 0.375, dtype=jnp.float64)

    coarse = contract_densities(rho, len(shape))

    np.testing.assert_allclose(
        coarse,
        0.375,
        rtol=0.0,
        atol=0.0,
    )


def test_contract_densities_3d_partial_boundary_window():
    rho = jnp.array(
        [
            [[1.0]],
            [[3.0]],
            [[7.0]],
        ],
        dtype=jnp.float64,
    )

    coarse = contract_densities(rho, 3)

    expected = np.array(
        [
            [[2.0]],
            [[7.0]],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        coarse,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_contract_densities_boolean_mask_becomes_fractional():
    mask = jnp.array(
        [
            [[True], [False]],
            [[True], [True]],
            [[False], [False]],
        ]
    )

    coarse = contract_densities(mask, 3)

    expected = np.array(
        [
            [[0.75]],
            [[0.0]],
        ],
        dtype=np.float32,
    )

    assert coarse.dtype == jnp.float32

    np.testing.assert_allclose(
        coarse,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_contract_densities_jittable():
    rho = jnp.arange(
        15,
        dtype=jnp.float64,
    ).reshape(3, 5)

    contract_jit = jax.jit(lambda x: contract_densities(x, 2))

    actual = contract_jit(rho)
    expected = contract_densities(rho, 2)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )
