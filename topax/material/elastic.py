import itertools
from typing import NamedTuple

import jax.numpy as jnp


class Elastic(NamedTuple):
    elastic_module: float
    poisson_ratio: float
    thickness: float


def elastic_n_dofs_node(n_dims):
    if n_dims not in (2, 3):
        raise NotImplementedError()
    return n_dims


def construct_elastic(
    elastic_module,
    poisson_ratio,
    thickness=None,
):
    return Elastic(
        elastic_module=elastic_module,
        poisson_ratio=poisson_ratio,
        thickness=thickness,
    )


def construct_local_stiffness(level, elastic):
    n_dims = level.n_dims
    if n_dims == 2:
        return construct_local_stiffness_2D(level, elastic)

    if n_dims == 3:
        return construct_local_stiffness_3D(level, elastic)

    raise NotImplementedError()


def construct_local_stiffness_2D(level, elastic):
    hx, hy = level.element_sizes

    E = elastic.elastic_module
    v = elastic.poisson_ratio
    t = elastic.thickness

    # Plane stress constitutive matrix
    D = (
        E
        / (1.0 - v**2)
        * jnp.array(
            [
                [1.0, v, 0.0],
                [v, 1.0, 0.0],
                [0.0, 0.0, (1.0 - v) / 2.0],
            ]
        )
    )

    # 2x2 Gauss quadrature
    g = 1.0 / jnp.sqrt(3.0)
    gauss_points = (-g, g)

    K = jnp.zeros(
        (8, 8),
        dtype=D.dtype,
    )

    # Axis-aligned rectangular mapping:
    #
    # x = hx/2 * xi
    # y = hy/2 * eta
    #
    # det(J) = hx * hy / 4
    det_jacobian = hx * hy / 4.0

    for xi in gauss_points:
        for eta in gauss_points:
            # Node ordering:
            #
            #   1 ----- 3
            #   |       |
            #   |       |
            #   0 ----- 2
            #
            # 0: (-1, -1)
            # 1: (-1, +1)
            # 2: (+1, -1)
            # 3: (+1, +1)

            dN_dxi = 0.25 * jnp.array(
                [
                    -(1.0 - eta),
                    -(1.0 + eta),
                    +(1.0 - eta),
                    +(1.0 + eta),
                ]
            )

            dN_deta = 0.25 * jnp.array(
                [
                    -(1.0 - xi),
                    +(1.0 - xi),
                    -(1.0 + xi),
                    +(1.0 + xi),
                ]
            )

            # Reference -> physical derivatives
            dN_dx = (2.0 / hx) * dN_dxi
            dN_dy = (2.0 / hy) * dN_deta

            # Component-major DOF ordering:
            #
            # [u0_x, u1_x, u2_x, u3_x,
            #  u0_y, u1_y, u2_y, u3_y]
            B = jnp.zeros(
                (3, 8),
                dtype=D.dtype,
            )

            B = B.at[0, :4].set(dN_dx)
            B = B.at[1, 4:].set(dN_dy)

            B = B.at[2, :4].set(dN_dy)
            B = B.at[2, 4:].set(dN_dx)

            K = K + (t * det_jacobian * (B.T @ D @ B))

    # TopaX operator layout:
    #
    # (output_local_dof,
    #  input_component,
    #  local_x,
    #  local_y)
    return K.reshape(8, 2, 2, 2)


def construct_local_stiffness_3D(level, elastic):
    hx, hy, hz = level.element_sizes

    E = elastic.elastic_module
    v = elastic.poisson_ratio

    lame_lambda = E * v / ((1.0 + v) * (1.0 - 2.0 * v))
    shear_modulus = E / (2.0 * (1.0 + v))

    # Three-dimensional isotropic constitutive matrix using engineering
    # shear strains [gamma_xy, gamma_yz, gamma_xz]. The 2D thickness is not
    # used for a true volume element.
    D = jnp.array(
        [
            [lame_lambda + 2.0 * shear_modulus, lame_lambda, lame_lambda, 0, 0, 0],
            [lame_lambda, lame_lambda + 2.0 * shear_modulus, lame_lambda, 0, 0, 0],
            [lame_lambda, lame_lambda, lame_lambda + 2.0 * shear_modulus, 0, 0, 0],
            [0, 0, 0, shear_modulus, 0, 0],
            [0, 0, 0, 0, shear_modulus, 0],
            [0, 0, 0, 0, 0, shear_modulus],
        ]
    )

    g = 1.0 / jnp.sqrt(3.0)
    gauss_points = (-g, g)

    # Node ordering matches C-order reshaping and elements2nodes_3D:
    #
    #   0: (-1, -1, -1)    1: (-1, -1, +1)
    #   2: (-1, +1, -1)    3: (-1, +1, +1)
    #   4: (+1, -1, -1)    5: (+1, -1, +1)
    #   6: (+1, +1, -1)    7: (+1, +1, +1)
    node_signs = jnp.array(
        tuple(
            itertools.product(
                (-1.0, 1.0),
                repeat=3,
            )
        ),
        dtype=D.dtype,
    )

    K = jnp.zeros(
        (24, 24),
        dtype=D.dtype,
    )

    # Axis-aligned rectangular mapping:
    #
    # x = hx/2 * xi, y = hy/2 * eta, z = hz/2 * zeta
    # det(J) = hx * hy * hz / 8
    det_jacobian = hx * hy * hz / 8.0

    for xi, eta, zeta in itertools.product(
        gauss_points,
        repeat=3,
    ):
        sx = node_signs[:, 0]
        sy = node_signs[:, 1]
        sz = node_signs[:, 2]

        dN_dxi = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
        dN_deta = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
        dN_dzeta = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)

        dN_dx = (2.0 / hx) * dN_dxi
        dN_dy = (2.0 / hy) * dN_deta
        dN_dz = (2.0 / hz) * dN_dzeta

        # Component-major DOF ordering:
        #
        # [ux0, ..., ux7, uy0, ..., uy7, uz0, ..., uz7]
        B = jnp.zeros(
            (6, 24),
            dtype=D.dtype,
        )

        B = B.at[0, 0:8].set(dN_dx)
        B = B.at[1, 8:16].set(dN_dy)
        B = B.at[2, 16:24].set(dN_dz)

        B = B.at[3, 0:8].set(dN_dy)
        B = B.at[3, 8:16].set(dN_dx)

        B = B.at[4, 8:16].set(dN_dz)
        B = B.at[4, 16:24].set(dN_dy)

        B = B.at[5, 0:8].set(dN_dz)
        B = B.at[5, 16:24].set(dN_dx)

        K = K + det_jacobian * (B.T @ D @ B)

    # TopaX operator layout:
    #
    # (output_local_dof,
    #  input_component,
    #  local_x,
    #  local_y,
    #  local_z)
    return K.reshape(24, 3, 2, 2, 2)
