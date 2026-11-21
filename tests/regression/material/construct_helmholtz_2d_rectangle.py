import math

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from topax.grid import Grid
from topax.material.helmholtz import (
    construct_helmholtz,
    construct_local_helmholtz,
    helmholtz_to_elemental,
    helmholtz_to_nodal,
)
from topax.operator import matrix_free_matvec


jax.config.update(
    "jax_enable_x64",
    True,
)


def _construct_helmholtz_2d_reference(
    h,
    filter_radius,
    thickness,
):
    """Old hard-coded square Q4 Helmholtz matrix used as regression oracle."""
    r = filter_radius / (2.0 * math.sqrt(3.0))

    K1 = jnp.array(
        [
            [4.0, -1.0, -1.0, -2.0],
            [-1.0, 4.0, -2.0, -1.0],
            [-1.0, -2.0, 4.0, -1.0],
            [-2.0, -1.0, -1.0, 4.0],
        ]
    )

    K2 = jnp.array(
        [
            [4.0, 2.0, 2.0, 1.0],
            [2.0, 4.0, 1.0, 2.0],
            [2.0, 1.0, 4.0, 2.0],
            [1.0, 2.0, 2.0, 4.0],
        ]
    )

    K = r**2 / 6.0 * K1 + h**2 / 36.0 * K2
    K = thickness * K

    return einops.rearrange(
        K,
        "oc (x y) -> oc 1 x y",
        x=2,
        y=2,
    )


def _flatten_local_helmholtz(
    local_helmholtz,
):
    return einops.rearrange(
        local_helmholtz,
        "oc ic x y -> oc (ic x y)",
    )


@pytest.mark.parametrize(
    "h",
    [
        0.2,
        1.0,
        7.3,
    ],
)
@pytest.mark.parametrize(
    "filter_radius, thickness",
    [
        (0.05, 1.0),
        (0.2, 0.01),
        (1.0, 2.3),
        (3.5, 0.7),
    ],
)
def test_construct_helmholtz_square_matches_hardcoded(
    h,
    filter_radius,
    thickness,
):
    sizes = np.array(
        [
            h,
            h,
        ],
        dtype=float,
    )

    elements = np.array(
        [
            1,
            1,
        ],
        dtype=int,
    )

    helmholtz = construct_helmholtz(
        filter_radius=filter_radius,
        thickness=thickness,
    )

    grid = Grid(
        n_levels=1,
        sizes=sizes,
        elements=elements,
    )

    local_helmholtz = construct_local_helmholtz(
        grid.fine_level,
        helmholtz,
    )

    K_reference = _construct_helmholtz_2d_reference(
        h=h,
        filter_radius=filter_radius,
        thickness=thickness,
    )

    np.testing.assert_allclose(
        local_helmholtz,
        K_reference,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "h",
    [
        0.2,
        1.0,
        7.3,
    ],
)
@pytest.mark.parametrize(
    "thickness",
    [
        0.01,
        1.0,
        2.3,
    ],
)
def test_helmholtz_to_nodal_square_matches_reference(
    h,
    thickness,
):
    sizes = np.array(
        [
            2.0 * h,
            3.0 * h,
        ],
        dtype=float,
    )

    elements = np.array(
        [
            2,
            3,
        ],
        dtype=int,
    )

    helmholtz = construct_helmholtz(
        filter_radius=1.0,
        thickness=thickness,
    )

    grid = Grid(
        n_levels=1,
        sizes=sizes,
        elements=elements,
    )

    level = grid.fine_level

    rho = jnp.arange(
        1,
        7,
        dtype=jnp.float64,
    ).reshape(
        2,
        3,
    )

    # Old square-element scaling:
    #
    #   rho_e * t * h^2 / 4
    #
    scaled = rho * thickness * h**2 / 4.0

    expected = jnp.zeros(
        (
            3,
            4,
        )
    )

    expected = expected.at[
        :-1,
        :-1,
    ].add(scaled)

    expected = expected.at[
        1:,
        :-1,
    ].add(scaled)

    expected = expected.at[
        :-1,
        1:,
    ].add(scaled)

    expected = expected.at[
        1:,
        1:,
    ].add(scaled)

    expected = expected[
        None,
        ...,
    ]

    actual = helmholtz_to_nodal(
        level,
        helmholtz,
        rho,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "sizes, elements",
    [
        (
            [6.0, 1.5],
            [3, 3],
        ),
        (
            [1.5, 6.0],
            [3, 3],
        ),
        (
            [4.0, 1.5],
            [4, 3],
        ),
    ],
    ids=[
        "hx=2.0-hy=0.5",
        "hx=0.5-hy=2.0",
        "hx=1.0-hy=0.5",
    ],
)
def test_helmholtz_rectangular_preserves_constant_density(
    sizes,
    elements,
):
    sizes = np.array(
        sizes,
        dtype=float,
    )

    elements = np.array(
        elements,
        dtype=int,
    )

    helmholtz = construct_helmholtz(
        filter_radius=1.0,
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

    # Constant elemental density.
    rho_value = 0.37

    rho = jnp.full(
        tuple(elements),
        rho_value,
        dtype=jnp.float64,
    )

    # No masked-out elements.
    rho_mask = jnp.ones_like(rho)

    # Assemble the Helmholtz RHS:
    #
    #     (-r^2 Laplacian + I) rho_tilde = rho
    #
    rhs = helmholtz_to_nodal(
        level,
        helmholtz,
        rho,
    )

    # -------------------------------------------------------------
    # Materialize the small matrix-free operator as a dense matrix.
    #
    # This is only for the test. Each column is A @ e_i.
    # -------------------------------------------------------------

    nodal_shape = tuple(level.nodes)

    n_nodes = level.n_nodes

    basis = jnp.eye(
        n_nodes,
        dtype=jnp.float64,
    )

    def matvec_flat(x):
        x = x.reshape(
            (
                1,
                *nodal_shape,
            )
        )

        return matrix_free_matvec(
            level,
            x,
            local_helmholtz,
            rho_mask,
        ).reshape(-1)

    K = jnp.stack(
        [matvec_flat(basis[:, i]) for i in range(n_nodes)],
        axis=1,
    )

    # Solve the complete discrete Helmholtz problem.
    rho_filtered_nodal = jnp.linalg.solve(
        K,
        rhs.reshape(-1),
    ).reshape(
        (
            1,
            *nodal_shape,
        )
    )

    # A constant field must remain constant under the Helmholtz filter.
    np.testing.assert_allclose(
        rho_filtered_nodal,
        rho_value,
        rtol=1e-11,
        atol=1e-11,
    )

    # Map the nodal solution back to elemental densities.
    rho_filtered = helmholtz_to_elemental(
        level,
        rho_filtered_nodal,
    )

    np.testing.assert_allclose(
        rho_filtered,
        rho,
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize(
    "hx, hy",
    [
        (2.0, 0.5),
        (0.5, 2.0),
        (1.7, 0.4),
    ],
)
@pytest.mark.parametrize(
    "filter_radius, thickness",
    [
        (1.0, 1.0),
        (0.6, 2.3),
    ],
)
def test_helmholtz_rectangular_linear_field_gradient_energy(
    hx,
    hy,
    filter_radius,
    thickness,
):
    sizes = np.array(
        [
            hx,
            hy,
        ],
        dtype=float,
    )

    elements = np.array(
        [
            1,
            1,
        ],
        dtype=int,
    )

    grid = Grid(
        n_levels=1,
        sizes=sizes,
        elements=elements,
    )

    level = grid.fine_level

    helmholtz = construct_helmholtz(
        filter_radius=filter_radius,
        thickness=thickness,
    )

    # Same element with r = 0 contains only the mass contribution.
    helmholtz_mass = construct_helmholtz(
        filter_radius=0.0,
        thickness=thickness,
    )

    local_helmholtz = construct_local_helmholtz(
        level,
        helmholtz,
    )

    local_helmholtz_mass = construct_local_helmholtz(
        level,
        helmholtz_mass,
    )

    K = _flatten_local_helmholtz(local_helmholtz)

    M = _flatten_local_helmholtz(local_helmholtz_mass)

    # Isolate:
    #
    #     t * r^2 ∫ grad(N)^T grad(N) dΩ
    #
    K_gradient = K - M

    # Node ordering:
    #
    #   1 ----- 3
    #   |       |
    #   |       |
    #   0 ----- 2
    #
    # Linear field increasing from 0 -> 1 in x:
    #
    #     0 ----- 1
    #     |       |
    #     0 ----- 1
    #
    linear_x = jnp.array(
        [
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        dtype=jnp.float64,
    )

    # Linear field increasing from 0 -> 1 in y:
    #
    #     1 ----- 1
    #     |       |
    #     0 ----- 0
    #
    linear_y = jnp.array(
        [
            0.0,
            1.0,
            0.0,
            1.0,
        ],
        dtype=jnp.float64,
    )

    energy_x = linear_x @ K_gradient @ linear_x

    energy_y = linear_y @ K_gradient @ linear_y

    r = filter_radius / (2.0 * np.sqrt(3.0))

    # For the x-linear field:
    #
    #     grad(rho) = [1/hx, 0]
    #
    # therefore
    #
    #     t r^2 ∫ |grad(rho)|² dΩ
    #       = t r² * hx*hy / hx²
    #       = t r² * hy/hx
    #
    expected_x = thickness * r**2 * hy / hx

    # Similarly:
    #
    #     grad(rho) = [0, 1/hy]
    #
    # gives
    #
    #     t r² * hx/hy
    #
    expected_y = thickness * r**2 * hx / hy

    np.testing.assert_allclose(
        energy_x,
        expected_x,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        energy_y,
        expected_y,
        rtol=1e-12,
        atol=1e-12,
    )

    # x- and y-gradients are orthogonal.
    cross_energy = linear_x @ K_gradient @ linear_y

    np.testing.assert_allclose(
        cross_energy,
        0.0,
        rtol=1e-12,
        atol=1e-12,
    )
