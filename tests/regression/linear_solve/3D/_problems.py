import numpy as np

import jax.numpy as jnp

import topax


SIZES = np.array([4.0, 0.2, 0.2])

ELASTIC_MODULE = 200 * 1e9
POISSON_RATIO = 0.3
THICKNESS = 1.0

SIMP_PENALTY = 3
RHO_MIN = 1e-9
RHO = 1.0


CONCENTRATED_FORCES = [
    [],
    [],
    [
        {
            "center": [1.0, 0.5, 0.0],
            "size": [0.0, 0.0, 0.0],
            "intensity": -10000.0,
        }
    ],
]

_FIXED_X_FACE = {
    "center": [0.0, 0.5, 0.5],
    "size": [0.0, 1.0, 1.0],
}

DIRICHLET_BOUNDARY_CONDITIONS = [
    [_FIXED_X_FACE],
    [_FIXED_X_FACE],
    [_FIXED_X_FACE],
]


def construct_linear_problem_3d(elements):
    elements = np.asarray(elements, dtype=int)

    elastic = topax.construct_elastic(
        ELASTIC_MODULE,
        POISSON_RATIO,
        THICKNESS,
    )

    grid = topax.construct_grid(
        n_levels=1,
        sizes=SIZES,
        elements=elements,
    )

    local_stiffness = topax.construct_local_stiffness(
        grid.fine_level,
        elastic,
    )

    restriction = topax.construct_restrictions(
        grid,
        elastic,
        DIRICHLET_BOUNDARY_CONDITIONS,
    )[0]

    forces = topax.construct_forces(
        grid,
        elastic,
        CONCENTRATED_FORCES,
    )

    rho = RHO * jnp.ones(
        tuple(elements),
        dtype=jnp.float64,
    )

    return grid, local_stiffness, restriction, forces, rho
