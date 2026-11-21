import numpy as np
import jax
import jax.numpy as jnp


def construct_heaviside(beta, eta):
    def heaviside_projection(rho):
        return (jnp.tanh(beta * (rho - eta)) + jnp.tanh(beta * eta)) / (
            jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
        )

    return heaviside_projection


def construct_conv_filter(sizes, elements, radius):
    dim = sizes.shape[0]

    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    if radius <= 0:
        raise ValueError("radius must be positive")

    element_size = np.asarray(sizes) / np.asarray(elements)

    # Number of element offsets needed in each direction.
    support = np.ceil(radius / element_size).astype(int)

    coords = [jnp.arange(-R + 1, R) * h for R, h in zip(support, element_size)]

    if dim == 2:
        dx, dy = jnp.meshgrid(
            coords[0],
            coords[1],
            indexing="ij",
        )

        dist = jnp.sqrt(dx**2 + dy**2)
        kernel = jnp.maximum(radius - dist, 0.0)

        dimension_numbers = (
            "NCHW",
            "OIHW",
            "NCHW",
        )

        strides = (1, 1)

    else:
        dx, dy, dz = jnp.meshgrid(
            coords[0],
            coords[1],
            coords[2],
            indexing="ij",
        )

        dist = jnp.sqrt(dx**2 + dy**2 + dz**2)
        kernel = jnp.maximum(radius - dist, 0.0)

        dimension_numbers = (
            "NCDHW",
            "OIDHW",
            "NCDHW",
        )

        strides = (1, 1, 1)

    def conv(x, kernel):
        x = jnp.asarray(x)
        kernel = jnp.asarray(
            kernel,
            dtype=x.dtype,
        )

        x = x[None, None, ...]
        kernel = kernel[None, None, ...]

        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=strides,
            padding="SAME",
            dimension_numbers=dimension_numbers,
        )

        return y[0, 0, ...]

    def conv_filter(rho, mask):
        rho_masked = rho * mask
        num = conv(rho_masked, kernel)
        denom = conv(mask.astype(kernel.dtype), kernel)
        rho_filtered = num / jnp.maximum(denom, 1e-12)

        return rho_filtered * mask

    return conv_filter
