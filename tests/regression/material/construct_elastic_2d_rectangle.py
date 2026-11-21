import einops
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from topax.grid import Grid
from topax.material.elastic import (
    construct_elastic,
    construct_local_stiffness,
)


jax.config.update(
    "jax_enable_x64",
    True,
)


def _construct_elastic_2d_reference(
    elastic_module,
    poisson_ratio,
    thickness,
):
    """Old hard-coded square Q4 stiffness used as a regression oracle."""
    E = elastic_module
    v = poisson_ratio
    t = thickness

    g11 = (v - 3) / (6 * v**2 - 6)
    g12 = -v / (6 * v**2 - 6)
    g13 = (v + 3) / (12 * v**2 - 12)
    g14 = -(v - 3) / (12 * v**2 - 12)
    g15 = -1 / (8 * v - 8)
    g17 = -(3 * v - 1) / (8 * v**2 - 8)

    g16 = -g17
    g18 = -g15

    K00 = jnp.array(
        [
            [[g11, g12], [g13, g14]],
            [[g12, g11], [g14, g13]],
            [[g13, g14], [g11, g12]],
            [[g14, g13], [g12, g11]],
        ]
    )

    K11 = jnp.array(
        [
            [[g11, g13], [g12, g14]],
            [[g13, g11], [g14, g12]],
            [[g12, g14], [g11, g13]],
            [[g14, g12], [g13, g11]],
        ]
    )

    K01 = jnp.array(
        [
            [[g15, g16], [g17, g18]],
            [[g17, g18], [g15, g16]],
            [[g16, g15], [g18, g17]],
            [[g18, g17], [g16, g15]],
        ]
    )

    K10 = einops.rearrange(
        K01,
        "(c1 c2) x y -> (x y) c1 c2",
        c1=2,
    )

    K0 = jnp.stack(
        [
            K00,
            K01,
        ],
        axis=1,
    )

    K1 = jnp.stack(
        [
            K10,
            K11,
        ],
        axis=1,
    )

    K = jnp.concat(
        [
            K0,
            K1,
        ],
        axis=0,
    )

    return E * t * K


@pytest.mark.parametrize(
    "h",
    [
        0.2,
        1.0,
        7.3,
    ],
)
@pytest.mark.parametrize(
    "E, nu, thickness",
    [
        (1.0, 0.3, 1.0),
        (210.0, 0.3, 0.01),
        (7.5, 0.15, 2.3),
        (123.0, 0.45, 0.7),
    ],
)
def test_construct_elastic_square_matches_hardcoded(
    h,
    E,
    nu,
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

    elastic = construct_elastic(
        elastic_module=E,
        poisson_ratio=nu,
        thickness=thickness,
    )

    grid = Grid(
        n_levels=1,
        sizes=sizes,
        elements=elements,
    )

    local_stiffness = construct_local_stiffness(
        grid.fine_level,
        elastic,
    )

    K_reference = _construct_elastic_2d_reference(
        elastic_module=E,
        poisson_ratio=nu,
        thickness=thickness,
    )

    np.testing.assert_allclose(
        local_stiffness,
        K_reference,
        rtol=1e-12,
        atol=1e-12,
    )
