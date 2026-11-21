import jax
import jax.numpy as jnp
import numpy as np
import pytest

from topax.material.density import construct_conv_filter


jax.config.update("jax_enable_x64", True)


def _construct_conv_filter_reference(radius):
    """Old index-space implementation for unit-square regression."""

    R = int(jnp.ceil(radius))

    coords = jnp.arange(
        -R + 1,
        R,
        dtype=jnp.float64,
    )

    dx, dy = jnp.meshgrid(
        coords,
        coords,
        indexing="ij",
    )

    dist = jnp.sqrt(dx**2 + dy**2)

    kernel = jnp.maximum(
        radius - dist,
        0.0,
    )

    def conv(x):
        x = x[None, None, ...]
        k = kernel[None, None, ...]

        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=k,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=(
                "NCHW",
                "OIHW",
                "NCHW",
            ),
        )

        return y[0, 0]

    def filter_(rho, mask):
        num = conv(rho * mask)
        denom = conv(mask.astype(rho.dtype))

        result = num / jnp.maximum(
            denom,
            1e-12,
        )

        return result * mask

    return filter_


@pytest.mark.parametrize(
    "radius",
    [
        1.5,
        2.5,
        3.5,
    ],
)
def test_conv_filter_unit_square_matches_old_implementation(
    radius,
):
    elements = np.array([7, 6])
    sizes = np.array([7.0, 6.0])

    rho = jnp.arange(
        42,
        dtype=jnp.float64,
    ).reshape(7, 6)

    rho = rho / rho.max()

    mask = jnp.ones_like(
        rho,
        dtype=bool,
    )

    # Include a hole so normalization with the mask
    # is tested as well.
    mask = mask.at[2, 3].set(False)
    mask = mask.at[5, 1].set(False)

    old_filter = _construct_conv_filter_reference(radius)

    new_filter = construct_conv_filter(
        sizes=sizes,
        elements=elements,
        radius=radius,
    )

    expected = old_filter(
        rho,
        mask,
    )

    actual = new_filter(
        rho,
        mask,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def _physical_conv_filter_reference(
    rho,
    mask,
    sizes,
    elements,
    radius,
):
    rho = np.asarray(rho)
    mask = np.asarray(mask)

    element_size = np.asarray(sizes) / np.asarray(elements)

    hx, hy = element_size

    result = np.zeros_like(
        rho,
        dtype=float,
    )

    nx, ny = rho.shape

    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                continue

            numerator = 0.0
            denominator = 0.0

            for k in range(nx):
                for m in range(ny):
                    if not mask[k, m]:
                        continue

                    dx = (i - k) * hx
                    dy = (j - m) * hy

                    distance = np.sqrt(dx**2 + dy**2)

                    weight = max(
                        radius - distance,
                        0.0,
                    )

                    numerator += weight * rho[k, m]

                    denominator += weight

            if denominator > 0:
                result[i, j] = numerator / denominator

    return result


@pytest.mark.parametrize(
    "sizes, elements, radius",
    [
        (
            [10.0, 2.5],
            [5, 5],
            2.5,
        ),  # hx=2.0, hy=0.5
        (
            [2.5, 10.0],
            [5, 5],
            2.5,
        ),  # hx=0.5, hy=2.0
        (
            [7.5, 2.0],
            [5, 4],
            2.2,
        ),  # hx=1.5, hy=0.5
    ],
)
def test_conv_filter_rectangular_matches_physical_reference(
    sizes,
    elements,
    radius,
):
    sizes = np.asarray(
        sizes,
        dtype=float,
    )

    elements = np.asarray(
        elements,
        dtype=int,
    )

    rho = jnp.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.4, 0.5, 1.0, 0.7, 0.8],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [0.8, 0.9, 1.0, 0.9, 0.8],
        ],
        dtype=jnp.float64,
    )

    # Trim if the parametrized mesh isn't 5x5.
    rho = rho[
        : elements[0],
        : elements[1],
    ]

    mask = jnp.ones_like(
        rho,
        dtype=bool,
    )

    # Also exercise masked normalization.
    mask = mask.at[1, 1].set(False)

    conv_filter = construct_conv_filter(
        sizes=sizes,
        elements=elements,
        radius=radius,
    )

    actual = conv_filter(
        rho,
        mask,
    )

    expected = _physical_conv_filter_reference(
        rho=rho,
        mask=mask,
        sizes=sizes,
        elements=elements,
        radius=radius,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
