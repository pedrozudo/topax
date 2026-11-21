import itertools
import numbers

import jax
import jax.numpy as jnp
import einops

from .simp import simp


DEFAULT_SLAB_SIZE = 12


def _make_unpack_out_channel_pattern(n_dims):
    left = "(c " + " ".join(f"x{i}c" for i in range(n_dims)) + ")"
    right = " ".join(f"x{i}" for i in range(n_dims))
    groups = " ".join(f"(x{i} x{i}c)" for i in range(n_dims))
    return f"{left} {right} -> c {groups}"


def _make_reduce_pattern(n_dims):
    groups = " ".join(f"(x{i} x{i}c)" for i in range(n_dims))
    axes = " ".join(f"x{i}" for i in range(n_dims))
    return f"c {groups} -> c {axes}"


def _matrix_free_matvec_local(level, x, kernel, weighing):
    """Apply the finite-element operator to one slab including its halos."""

    n_dims = level.n_dims
    corner_sizes = {f"x{i}c": 2 for i in range(n_dims)}
    result = jax.lax.conv_with_general_padding(
        x[jnp.newaxis, :],
        kernel,
        window_strides=(1,) * n_dims,
        padding="VALID",
        lhs_dilation=None,
        rhs_dilation=None,
    )[0]
    result = weighing * result
    result = einops.rearrange(
        result,
        _make_unpack_out_channel_pattern(n_dims),
        **corner_sizes,
    )
    result = jnp.pad(
        result,
        ((0, 0),) + ((1, 1),) * n_dims,
        constant_values=0.0,
    )
    result = einops.reduce(
        result,
        _make_reduce_pattern(n_dims),
        "sum",
        **corner_sizes,
    )
    return result


def _validate_slicing_parameter(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _resolve_slab_axis(slab_axis, spatial_shape):
    if slab_axis is None:
        return max(range(len(spatial_shape)), key=spatial_shape.__getitem__)
    if (
        isinstance(slab_axis, bool)
        or not isinstance(slab_axis, numbers.Integral)
    ):
        raise ValueError("slab_axis must be an integer or None.")
    slab_axis = int(slab_axis)
    if not 0 <= slab_axis < len(spatial_shape):
        raise ValueError(
            f"slab_axis must satisfy 0 <= axis < {len(spatial_shape)}."
        )
    return slab_axis


def matrix_free_matvec(
    level,
    x,
    kernel,
    weighing,
    restrictions=None,
    *,
    slab_size=DEFAULT_SLAB_SIZE,
    slab_axis=None,
    rematerialize=True,
):
    """Apply a 2D or 3D finite-element operator in bounded nodal slabs.

    Each slab owns ``slab_size`` output-node planes along ``slab_axis``. By
    default, the largest element dimension is selected to minimize the slab
    cross-section. Interior slabs include one adjacent element plane and one
    input-node plane on either side. Boundary slabs use only the available
    one-sided halo, avoiding padded copies of the complete fields. Slab
    results are written directly into the global result instead of first
    constructing a stacked array of all slabs. ``rematerialize`` bounds
    reverse-mode residual storage by recomputing slab bodies in the pullback.
    """

    n_dims = level.n_dims
    if n_dims not in (2, 3):
        raise NotImplementedError(
            "Sliced matrix-free matvec supports only 2D and 3D."
        )

    slab_size = _validate_slicing_parameter(slab_size, "slab_size")
    if not isinstance(rematerialize, bool):
        raise ValueError("rematerialize must be a Python boolean.")

    x = jnp.asarray(x)
    weighing = jnp.asarray(weighing)
    expected_spatial_shape = tuple(map(int, level.nodes))
    expected_weighing_shape = tuple(map(int, level.elements))
    slab_axis = _resolve_slab_axis(slab_axis, expected_weighing_shape)
    if x.ndim != n_dims + 1 or x.shape[1:] != expected_spatial_shape:
        raise ValueError(
            f"Expected a component-first {n_dims}D nodal field with spatial shape "
            f"{expected_spatial_shape}, got {x.shape}."
        )
    if weighing.shape != expected_weighing_shape:
        raise ValueError(
            f"Expected weighing shape {expected_weighing_shape}, "
            f"got {weighing.shape}."
        )
    if restrictions is not None and restrictions.shape != x.shape:
        raise ValueError(
            f"Expected restrictions shape {x.shape}, got {restrictions.shape}."
        )

    n_nodes_axis = expected_spatial_shape[slab_axis]
    field_axis = slab_axis + 1
    slab_size = min(slab_size, n_nodes_axis)
    n_slabs = (n_nodes_axis + slab_size - 1) // slab_size

    def apply_owned_slab(
        start,
        owned_size,
        *,
        has_left_halo,
        has_right_halo,
    ):
        local_start = start - int(has_left_halo)
        n_local_elements = (
            owned_size
            - 1
            + int(has_left_halo)
            + int(has_right_halo)
        )
        x_local = jax.lax.dynamic_slice_in_dim(
            x,
            local_start,
            n_local_elements + 1,
            axis=field_axis,
        )
        weighing_local = jax.lax.dynamic_slice_in_dim(
            weighing,
            local_start,
            n_local_elements,
            axis=slab_axis,
        )
        result_local = _matrix_free_matvec_local(
            level,
            x_local,
            kernel,
            weighing_local,
        )
        return jax.lax.dynamic_slice_in_dim(
            result_local,
            int(has_left_halo),
            owned_size,
            axis=field_axis,
        )

    if n_slabs == 1:
        result = apply_owned_slab(
            0,
            n_nodes_axis,
            has_left_halo=False,
            has_right_halo=False,
        )
    else:
        def apply_first_slab():
            return apply_owned_slab(
                0,
                slab_size,
                has_left_halo=False,
                has_right_halo=True,
            )

        def apply_interior_slab(start):
            return apply_owned_slab(
                start,
                slab_size,
                has_left_halo=True,
                has_right_halo=True,
            )

        last_start = (n_slabs - 1) * slab_size
        last_size = n_nodes_axis - last_start

        def apply_last_slab():
            return apply_owned_slab(
                last_start,
                last_size,
                has_left_halo=True,
                has_right_halo=False,
            )

        if rematerialize:
            apply_first_slab = jax.checkpoint(
                apply_first_slab,
                prevent_cse=False,
            )
            apply_interior_slab = jax.checkpoint(
                apply_interior_slab,
                prevent_cse=False,
            )
            apply_last_slab = jax.checkpoint(
                apply_last_slab,
                prevent_cse=False,
            )

        result = jnp.zeros_like(x)
        first_result = apply_first_slab()
        result = jax.lax.dynamic_update_slice_in_dim(
            result,
            first_result,
            0,
            axis=field_axis,
        )

        def write_interior_slab(slab_index, current_result):
            start = slab_index * slab_size
            local_result = apply_interior_slab(start)
            return jax.lax.dynamic_update_slice_in_dim(
                current_result,
                local_result,
                start,
                axis=field_axis,
            )

        if n_slabs > 2:
            result = jax.lax.fori_loop(
                1,
                n_slabs - 1,
                write_interior_slab,
                result,
            )
        last_result = apply_last_slab()
        result = jax.lax.dynamic_update_slice_in_dim(
            result,
            last_result,
            last_start,
            axis=field_axis,
        )
    if restrictions is not None:
        result = result * restrictions
    return result


def _element_strain_energy_local(x, kernel):
    """Compute element energies for one complete grid or spatial slab."""

    n_dims = x.ndim - 1
    n_components = x.shape[0]
    n_local_dofs = n_components * 2**n_dims
    element_shape = tuple(size - 1 for size in x.shape[1:])
    local_values = []
    for component in range(n_components):
        for corner in itertools.product((0, 1), repeat=n_dims):
            local_slice = tuple(
                slice(offset, offset + size)
                for offset, size in zip(corner, element_shape)
            )
            local_values.append(x[(component,) + local_slice])
    local_values = jnp.stack(local_values)
    element_matrix = kernel.reshape(n_local_dofs, n_local_dofs)
    return jnp.einsum(
        "i...,ij,j...->...",
        jnp.conj(local_values),
        element_matrix,
        local_values,
    ).real


def _validate_element_strain_energy_inputs(level, x, kernel, restrictions):
    n_dims = level.n_dims
    if n_dims not in (2, 3):
        raise NotImplementedError(
            "Element strain energy supports only 2D and 3D."
        )

    x = jnp.asarray(x)
    kernel = jnp.asarray(kernel)
    expected_spatial_shape = tuple(map(int, level.nodes))
    if x.ndim != n_dims + 1 or x.shape[1:] != expected_spatial_shape:
        raise ValueError(
            f"Expected a component-first {n_dims}D nodal field with spatial shape "
            f"{expected_spatial_shape}, got {x.shape}."
        )
    if restrictions is not None and restrictions.shape != x.shape:
        raise ValueError(
            f"Expected restrictions shape {x.shape}, got {restrictions.shape}."
        )

    n_components = x.shape[0]
    n_local_dofs = n_components * level.n_nodes_element
    expected_kernel_shape = (
        n_local_dofs,
        n_components,
        *((2,) * n_dims),
    )
    if kernel.shape != expected_kernel_shape:
        raise ValueError(
            f"Expected local operator shape {expected_kernel_shape}, "
            f"got {kernel.shape}."
        )
    return x, kernel


def element_strain_energy(
    level,
    x,
    kernel,
    restrictions=None,
    *,
    slab_size=DEFAULT_SLAB_SIZE,
    slab_axis=None,
):
    """Compute element strain energies in bounded spatial slabs.

    Only the element-local displacement and ``K_e @ u_e`` arrays for the
    current slab are constructed.
    """

    x, kernel = _validate_element_strain_energy_inputs(
        level,
        x,
        kernel,
        restrictions,
    )
    slab_size = _validate_slicing_parameter(slab_size, "slab_size")

    element_shape = tuple(map(int, level.elements))
    slab_axis = _resolve_slab_axis(slab_axis, element_shape)
    field_axis = slab_axis + 1
    n_elements_axis = element_shape[slab_axis]
    slab_size = min(slab_size, n_elements_axis)
    n_full_slabs, tail_size = divmod(n_elements_axis, slab_size)

    def compute_slab(start, size):
        x_local = jax.lax.dynamic_slice_in_dim(
            x,
            start,
            size + 1,
            axis=field_axis,
        )
        if restrictions is not None:
            restrictions_local = jax.lax.dynamic_slice_in_dim(
                restrictions,
                start,
                size + 1,
                axis=field_axis,
            )
            x_local = x_local * restrictions_local
        return _element_strain_energy_local(x_local, kernel)

    result = jnp.zeros(element_shape, dtype=x.real.dtype)

    def write_full_slab(slab_index, current_result):
        start = slab_index * slab_size
        slab = compute_slab(start, slab_size)
        return jax.lax.dynamic_update_slice_in_dim(
            current_result,
            slab,
            start,
            axis=slab_axis,
        )

    result = jax.lax.fori_loop(
        0,
        n_full_slabs,
        write_full_slab,
        result,
    )
    if tail_size:
        tail_start = n_full_slabs * slab_size
        tail = compute_slab(tail_start, tail_size)
        result = jax.lax.dynamic_update_slice_in_dim(
            result,
            tail,
            tail_start,
            axis=slab_axis,
        )
    return result


def matrix_free_diagonal(level, kernel, weighing):
    """Assemble the diagonal of a matrix-free finite-element operator."""

    n_dims = level.n_dims
    n_channels = kernel.shape[1]
    n_dofs_element = n_channels * 2**n_dims
    element_matrix = kernel.reshape(n_dofs_element, n_dofs_element)
    local_diagonal = jnp.diag(element_matrix).reshape(
        (n_channels,) + (2,) * n_dims
    )
    diagonal = jnp.zeros(
        (n_channels, *level.nodes),
        dtype=kernel.dtype,
    )
    for corner in itertools.product((0, 1), repeat=n_dims):
        coefficient = local_diagonal[(slice(None),) + corner]
        coefficient = coefficient.reshape((n_channels,) + (1,) * n_dims)
        contribution = weighing[None, ...] * coefficient
        padding = ((0, 0),) + tuple((index, 1 - index) for index in corner)
        diagonal = diagonal + jnp.pad(contribution, padding)
    return diagonal


def physics_weighing(rho, simp_penalty, rho_min):
    return simp(rho, simp_penalty, rho_min)
