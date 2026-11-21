import numpy as np

import jax.numpy as jnp

import topax


# =============================================================================
# Physical problem
# =============================================================================

SIZES = np.array(
    [
        1.0,
        1.0,
    ]
)

ELASTIC_MODULE = 1.0
POISSON_RATIO = 0.3
THICKNESS = 1.0

SIMP_PENALTY = 3
RHO_MIN = 1e-9

RHO = 0.5


CONCENTRATED_FORCES = [
    [],
    [
        {
            "center": [
                SIZES[0],
                0.0,
            ],
            "size": [
                0,
                0,
            ],
            "intensity": -1.0,
        }
    ],
]


DIRICHLET_BOUNDARY_CONDITIONS = [
    [
        {
            "center": [
                0.0,
                SIZES[1] / 2,
            ],
            "size": [
                0,
                SIZES[1],
            ],
        },
    ],
    [
        {
            "center": [
                0.0,
                SIZES[1] / 2,
            ],
            "size": [
                0,
                SIZES[1],
            ],
        },
    ],
]


# =============================================================================
# Construction
# =============================================================================


def construct_linear_problem(
    elements,
    n_levels,
):
    elements = np.asarray(
        elements,
        dtype=int,
    )

    elastic = topax.construct_elastic(
        ELASTIC_MODULE,
        POISSON_RATIO,
        THICKNESS,
    )

    grid = topax.construct_grid(
        n_levels,
        SIZES,
        elements,
    )

    local_stiffnesses = tuple(
        topax.construct_local_stiffness(
            level,
            elastic,
        )
        for level in grid.levels
    )

    restrictions = topax.construct_restrictions(
        grid,
        elastic,
        DIRICHLET_BOUNDARY_CONDITIONS,
    )

    forces = topax.construct_forces(
        grid,
        elastic,
        CONCENTRATED_FORCES,
    )

    rho = RHO * jnp.ones(
        elements,
        dtype=jnp.float64,
    )

    return (
        elastic,
        grid,
        local_stiffnesses,
        restrictions,
        forces,
        rho,
    )
