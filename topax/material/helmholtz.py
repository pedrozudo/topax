import itertools
import math
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
import einops


def construct_helmholtz(filter_radius, thickness):
    return Helmholtz(filter_radius=filter_radius, thickness=thickness)


class Helmholtz(NamedTuple):
    filter_radius: float
    thickness: float


def helmholtz_n_dofs_node(n_dims):
    if n_dims not in (2, 3):
        raise NotImplementedError()
    return 1


def construct_local_helmholtz(level, helmholtz):
    if level.n_dims == 2:
        return construct_helmholtz_2D(
            level.element_sizes,
            helmholtz.filter_radius,
            helmholtz.thickness,
        )

    if level.n_dims == 3:
        return construct_helmholtz_3D(
            level.element_sizes,
            helmholtz.filter_radius,
        )

    raise NotImplementedError()


def helmholtz_to_nodal(level, helmholtz, rho):
    reduce_window_dimensions_to_nodal = (2,) * level.n_dims
    reduce_window_strides_to_nodal = (1,) * level.n_dims
    reduce_window_padding_to_nodal = ((1, 1),) * level.n_dims
    element_measure = jnp.prod(level.element_sizes)
    if level.n_dims == 2:
        element_measure = helmholtz.thickness * element_measure
    rho = rho * element_measure / 2**level.n_dims
    rho_nodal = jax.lax.reduce_window(
        operand=rho,
        init_value=np.array(0.0, dtype=rho.dtype).item(),
        computation=jax.lax.add,
        window_dimensions=reduce_window_dimensions_to_nodal,
        window_strides=reduce_window_strides_to_nodal,
        padding=reduce_window_padding_to_nodal,
    )

    return rho_nodal[None, ...]


def helmholtz_to_elemental(level, rho):
    reduce_window_dimensions_to_elemental = (1,) + (2,) * level.n_dims
    reduce_window_strides_to_elemental = (1,) + (1,) * level.n_dims
    reduce_window_padding_to_elemental = ((0, 0),) + ((0, 0),) * level.n_dims
    rho = rho / 2**level.n_dims
    rho_elemental = jax.lax.reduce_window(
        operand=rho,
        init_value=np.array(0.0, dtype=rho.dtype).item(),
        computation=jax.lax.add,
        window_dimensions=reduce_window_dimensions_to_elemental,
        window_strides=reduce_window_strides_to_elemental,
        padding=reduce_window_padding_to_elemental,
    )
    return rho_elemental[0, ...]


def construct_helmholtz_2D(element_size, filter_radius, thickness):
    hx, hy = element_size
    r = filter_radius / (2.0 * math.sqrt(3.0))
    g = 1.0 / math.sqrt(3.0)
    gauss_points = (-g, g)
    K = jnp.zeros((4, 4))

    det_jacobian = hx * hy / 4.0

    for xi in gauss_points:
        for eta in gauss_points:
            N = 0.25 * jnp.array(
                [
                    (1.0 - xi) * (1.0 - eta),
                    (1.0 - xi) * (1.0 + eta),
                    (1.0 + xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 + eta),
                ]
            )

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

            dN_dx = (2.0 / hx) * dN_dxi
            dN_dy = (2.0 / hy) * dN_deta
            gradient = jnp.outer(dN_dx, dN_dx) + jnp.outer(dN_dy, dN_dy)
            mass = jnp.outer(N, N)

            K = K + (thickness * det_jacobian * (r**2 * gradient + mass))

    return einops.rearrange(K, "oc (x y) -> oc 1 x y", x=2, y=2)

def construct_helmholtz_3D(element_size, filter_radius):
    hx, hy, hz = element_size
    r = filter_radius / (2.0 * math.sqrt(3.0))
    g = 1.0 / math.sqrt(3.0)
    gauss_points = (-g, g)
    node_signs = jnp.array(tuple(itertools.product((-1.0, 1.0), repeat=3)))
    K = jnp.zeros((8, 8))

    det_jacobian = hx * hy * hz / 8.0

    for xi, eta, zeta in itertools.product(gauss_points, repeat=3):
        sx = node_signs[:, 0]
        sy = node_signs[:, 1]
        sz = node_signs[:, 2]

        N = 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta)
        dN_dx = (2.0 / hx) * 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
        dN_dy = (2.0 / hy) * 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
        dN_dz = (2.0 / hz) * 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)

        gradient = (
            jnp.outer(dN_dx, dN_dx)
            + jnp.outer(dN_dy, dN_dy)
            + jnp.outer(dN_dz, dN_dz)
        )
        mass = jnp.outer(N, N)
        K = K + det_jacobian * (r**2 * gradient + mass)

    return einops.rearrange(K, "oc (x y z) -> oc 1 x y z", x=2, y=2, z=2)
