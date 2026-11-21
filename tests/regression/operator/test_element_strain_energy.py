import jax
import jax.numpy as jnp
import numpy as np
import pytest

import topax


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("sizes", "elements"),
    [
        ((2.0, 1.0), (3, 2)),
        ((2.0, 1.0, 0.5), (3, 2, 2)),
    ],
)
def test_element_strain_energy_matches_matrix_free_quadratic_form(
    sizes,
    elements,
):
    grid = topax.construct_grid(1, np.asarray(sizes), np.asarray(elements))
    level = grid.fine_level
    elastic = topax.construct_elastic(1.7, 0.27, thickness=0.8)
    local_stiffness = topax.construct_local_stiffness(level, elastic)
    displacement = jnp.sin(
        jnp.arange(level.n_dims * level.n_nodes, dtype=jnp.float64) / 7.0
    ).reshape((level.n_dims, *map(int, level.nodes)))
    restrictions = jnp.ones_like(displacement).at[:, 0].set(0.0)
    displacement = displacement * restrictions
    rho = jnp.linspace(
        0.2,
        0.9,
        level.n_elements,
        dtype=jnp.float64,
    ).reshape(tuple(elements))
    weighing = topax.operator.physics_weighing(rho, 3.0, 1e-6)

    strain_energy = topax.operator.element_strain_energy(
        level,
        displacement,
        local_stiffness,
        restrictions,
    )
    direct_value = jnp.vdot(weighing, strain_energy)
    assembled_value = jnp.vdot(
        displacement * restrictions,
        topax.operator.matrix_free_matvec(
            level,
            displacement,
            local_stiffness,
            weighing,
            restrictions,
        ),
    )

    assert strain_energy.shape == tuple(elements)
    np.testing.assert_allclose(direct_value, assembled_value, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("sizes", "elements", "slab_size", "slab_axis"),
    [
        ((2.0, 1.0), (5, 3), 2, 0),
        ((2.0, 1.0), (3, 5), 2, 1),
        ((2.0, 1.0, 0.5), (5, 3, 2), 2, 0),
        ((2.0, 1.0, 0.5), (3, 5, 2), 2, 1),
        ((2.0, 1.0, 0.5), (3, 2, 5), 2, 2),
    ],
)
@pytest.mark.parametrize("with_restrictions", [False, True])
def test_element_strain_energy_matches_single_slab_under_jit(
    sizes,
    elements,
    slab_size,
    slab_axis,
    with_restrictions,
):
    grid = topax.construct_grid(1, np.asarray(sizes), np.asarray(elements))
    level = grid.fine_level
    elastic = topax.construct_elastic(1.7, 0.27, thickness=0.8)
    local_stiffness = topax.construct_local_stiffness(level, elastic)
    displacement = jnp.sin(
        jnp.arange(level.n_dims * level.n_nodes, dtype=jnp.float64) / 7.0
    ).reshape((level.n_dims, *map(int, level.nodes)))
    restrictions = jnp.ones_like(displacement).at[:, 0].set(0.0)
    if not with_restrictions:
        restrictions = None

    expected = topax.operator.element_strain_energy(
        level,
        displacement,
        local_stiffness,
        restrictions,
        slab_size=max(elements),
        slab_axis=slab_axis,
    )
    actual = jax.jit(
        lambda value: topax.operator.element_strain_energy(
            level,
            value,
            local_stiffness,
            restrictions,
            slab_size=slab_size,
            slab_axis=slab_axis,
        )
    )(displacement)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_element_energy_simp_gradient_matches_matrix_free_ad():
    grid = topax.construct_grid(1, np.ones(3), np.asarray((2, 2, 2)))
    level = grid.fine_level
    elastic = topax.construct_elastic(1.0, 0.3)
    local_stiffness = topax.construct_local_stiffness(level, elastic)
    displacement = jnp.cos(
        jnp.arange(3 * level.n_nodes, dtype=jnp.float64) / 5.0
    ).reshape((3, *map(int, level.nodes)))
    rho = jnp.linspace(0.3, 0.8, level.n_elements).reshape((2, 2, 2))
    strain_energy = topax.operator.element_strain_energy(
        level,
        displacement,
        local_stiffness,
    )

    direct_gradient = jax.grad(
        lambda density: jnp.vdot(
            topax.operator.physics_weighing(density, 3.0, 1e-6),
            strain_energy,
        )
    )(rho)
    matrix_free_gradient = jax.grad(
        lambda density: jnp.vdot(
            displacement,
            topax.operator.matrix_free_matvec(
                level,
                displacement,
                local_stiffness,
                topax.operator.physics_weighing(density, 3.0, 1e-6),
            ),
        )
    )(rho)

    np.testing.assert_allclose(
        direct_gradient,
        matrix_free_gradient,
        rtol=1e-12,
        atol=1e-12,
    )
