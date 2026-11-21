import itertools

import jax
import jax.numpy as jnp
import numpy as np

from topax.grid import Grid
from topax.material.elastic import construct_elastic, construct_local_stiffness
from topax.operator import matrix_free_matvec


jax.config.update(
    "jax_enable_x64",
    True,
)


def _construct_element(
    sizes=(2.0, 3.0, 4.0),
    elastic_module=10.0,
    poisson_ratio=0.3,
    thickness=1.0,
):
    grid = Grid(
        n_levels=1,
        sizes=sizes,
        elements=[1, 1, 1],
    )

    elastic = construct_elastic(
        elastic_module=elastic_module,
        poisson_ratio=poisson_ratio,
        thickness=thickness,
    )

    local_stiffness = construct_local_stiffness(
        grid.fine_level,
        elastic,
    )

    return grid.fine_level, local_stiffness


def _node_coordinates(element_sizes):
    hx, hy, hz = element_sizes

    return np.array(
        tuple(
            itertools.product(
                (0.0, hx),
                (0.0, hy),
                (0.0, hz),
            )
        )
    )


def _construct_constitutive_matrix(elastic_module, poisson_ratio):
    E = elastic_module
    v = poisson_ratio

    lame_lambda = E * v / ((1.0 + v) * (1.0 - 2.0 * v))
    shear_modulus = E / (2.0 * (1.0 + v))

    return np.array(
        [
            [lame_lambda + 2.0 * shear_modulus, lame_lambda, lame_lambda, 0, 0, 0],
            [lame_lambda, lame_lambda + 2.0 * shear_modulus, lame_lambda, 0, 0, 0],
            [lame_lambda, lame_lambda, lame_lambda + 2.0 * shear_modulus, 0, 0, 0],
            [0, 0, 0, shear_modulus, 0, 0],
            [0, 0, 0, 0, shear_modulus, 0],
            [0, 0, 0, 0, 0, shear_modulus],
        ]
    )


def test_hexahedral_stiffness_is_symmetric_with_six_rigid_body_modes():
    level, local_stiffness = _construct_element()

    assert local_stiffness.shape == (24, 3, 2, 2, 2)

    K = np.asarray(local_stiffness).reshape(24, 24)

    np.testing.assert_allclose(
        K,
        K.T,
        rtol=0.0,
        atol=1e-12,
    )

    coordinates = _node_coordinates(level.element_sizes)
    rigid_body_modes = []

    for component in range(3):
        translation = np.zeros((3, 8))
        translation[component, :] = 1.0
        rigid_body_modes.append(translation.reshape(-1))

    for rotation_axis in np.eye(3):
        rotation = np.cross(
            np.broadcast_to(rotation_axis, coordinates.shape),
            coordinates,
        ).T
        rigid_body_modes.append(rotation.reshape(-1))

    residuals = K @ np.stack(rigid_body_modes, axis=1)

    np.testing.assert_allclose(
        residuals,
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    eigenvalues = np.linalg.eigvalsh(K)

    np.testing.assert_allclose(
        eigenvalues[:6],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert eigenvalues[6] > 0.0


def test_hexahedral_stiffness_reproduces_affine_strain_energy():
    sizes = np.array([1.7, 0.8, 2.3])
    elastic_module = 7.5
    poisson_ratio = 0.27

    level, local_stiffness = _construct_element(
        sizes=sizes,
        elastic_module=elastic_module,
        poisson_ratio=poisson_ratio,
    )

    displacement_gradient = np.array(
        [
            [0.20, 0.05, -0.02],
            [0.03, -0.10, 0.04],
            [0.01, -0.06, 0.15],
        ]
    )

    coordinates = _node_coordinates(level.element_sizes)
    displacement = displacement_gradient @ coordinates.T

    engineering_strain = np.array(
        [
            displacement_gradient[0, 0],
            displacement_gradient[1, 1],
            displacement_gradient[2, 2],
            displacement_gradient[0, 1] + displacement_gradient[1, 0],
            displacement_gradient[1, 2] + displacement_gradient[2, 1],
            displacement_gradient[0, 2] + displacement_gradient[2, 0],
        ]
    )

    D = _construct_constitutive_matrix(
        elastic_module,
        poisson_ratio,
    )

    expected = np.prod(sizes) * engineering_strain @ D @ engineering_strain

    K = np.asarray(local_stiffness).reshape(24, 24)
    actual = displacement.reshape(-1) @ K @ displacement.reshape(-1)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_hexahedral_layout_matches_matrix_free_operator():
    level, local_stiffness = _construct_element(
        sizes=(1.2, 0.9, 1.7),
    )

    displacement = jnp.linspace(
        -0.7,
        1.1,
        24,
        dtype=jnp.float64,
    ).reshape(3, 2, 2, 2)

    actual = matrix_free_matvec(
        level,
        displacement,
        local_stiffness,
        jnp.ones((1, 1, 1), dtype=jnp.float64),
    ).reshape(-1)
    expected = local_stiffness.reshape(24, 24) @ displacement.reshape(-1)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_hexahedral_stiffness_does_not_use_2d_thickness():
    _, stiffness_1 = _construct_element(thickness=1.0)
    _, stiffness_2 = _construct_element(thickness=7.0)

    np.testing.assert_allclose(
        stiffness_1,
        stiffness_2,
        rtol=0.0,
        atol=0.0,
    )
