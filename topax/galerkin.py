"""Building blocks for a Galerkin elasticity hierarchy.

The routines in this module establish the algebra used by the multigrid
hierarchy:

    G_s = Q_s.T @ K_0 @ Q_s

and provide the heterogeneous element-matrix operators used after
materialization.
"""

import itertools
import numbers

import einops
import jax
import jax.numpy as jnp
import numpy as np

from .grid import Level
from .sparse import construct_elements2dofs, construct_elements2nodes


# The measured L1/L2 winners both process 2**19 subcell contributions per
# chunk, keeping contraction work bounded as the Galerkin level increases.
_GALERKIN_CHUNK_SUBCELL_BUDGET = 2**19
DEFAULT_GALERKIN_SLAB_SIZE = 4


def _validate_n_dims(n_dims):
    if n_dims not in (2, 3):
        raise NotImplementedError(
            f"Galerkin elasticity supports only 2D and 3D, got {n_dims}D."
        )
    return int(n_dims)


def _validate_n_subcells_per_axis(n_subcells_per_axis):
    if isinstance(n_subcells_per_axis, bool) or not isinstance(
        n_subcells_per_axis,
        numbers.Integral,
    ):
        raise ValueError("n_subcells_per_axis must be a positive integer.")
    if n_subcells_per_axis < 1:
        raise ValueError("n_subcells_per_axis must be a positive integer.")
    return int(n_subcells_per_axis)


def _infer_n_subcells_per_axis(n_subcells, n_dims):
    n = 1
    while n**n_dims < n_subcells:
        n += 1
    if n**n_dims != n_subcells:
        raise ValueError(
            "The number of Galerkin operators must be a perfect "
            f"{n_dims}-dimensional power."
        )
    return n


def construct_nodal_interpolation_matrices(
    n_subcells_per_axis,
    *,
    n_dims=3,
    dtype=jnp.float64,
):
    """Construct scalar-Q1 interpolation matrices for all fine subcells.

    Fine subcells and their local nodes follow Topax's C-order convention:
    the last spatial axis varies fastest. Each matrix maps a coarse
    macroelement's nodal values to one fine subcell's nodal values.
    """

    n_dims = _validate_n_dims(n_dims)
    n = _validate_n_subcells_per_axis(n_subcells_per_axis)
    local_nodes = np.asarray(
        tuple(itertools.product((0.0, 1.0), repeat=n_dims)),
        dtype=np.float64,
    )
    coarse_node_signs = 2.0 * local_nodes - 1.0
    subcells = np.asarray(
        tuple(itertools.product(range(n), repeat=n_dims)),
        dtype=np.float64,
    )

    # Fine-subcell nodes in macroelement reference coordinates [-1, 1]^d.
    points = -1.0 + 2.0 * (
        subcells[:, None, :] + local_nodes[None, :, :]
    ) / n
    basis_values = 0.5 * (
        1.0
        + points[:, :, None, :] * coarse_node_signs[None, None, :, :]
    )
    interpolation = np.prod(basis_values, axis=-1)
    return jnp.asarray(interpolation, dtype=dtype)


def construct_elastic_interpolation_matrices(
    n_subcells_per_axis,
    *,
    n_dims=3,
    dtype=jnp.float64,
):
    """Lift scalar interpolation to component-major elasticity DOFs."""

    n_dims = _validate_n_dims(n_dims)
    nodal_interpolation = construct_nodal_interpolation_matrices(
        n_subcells_per_axis,
        n_dims=n_dims,
        dtype=dtype,
    )
    identity = jnp.eye(n_dims, dtype=nodal_interpolation.dtype)
    return jax.vmap(lambda q: jnp.kron(identity, q))(nodal_interpolation)


def construct_galerkin_local_operators(
    fine_local_stiffness,
    n_subcells_per_axis,
    *,
    n_dims=3,
):
    """Construct every fixed Q_s.T @ K_0 @ Q_s local operator."""

    n_dims = _validate_n_dims(n_dims)
    n_dofs_element = n_dims * 2**n_dims
    fine_local_stiffness = jnp.asarray(fine_local_stiffness)
    if fine_local_stiffness.size != n_dofs_element**2:
        raise ValueError(
            f"Expected a {n_dofs_element} x {n_dofs_element} "
            f"{n_dims}D elasticity matrix."
        )
    fine_local_stiffness = fine_local_stiffness.reshape(
        n_dofs_element,
        n_dofs_element,
    )
    interpolation = construct_elastic_interpolation_matrices(
        n_subcells_per_axis,
        n_dims=n_dims,
        dtype=fine_local_stiffness.dtype,
    )
    return jax.vmap(
        lambda q: q.T @ fine_local_stiffness @ q,
    )(interpolation)


def group_fine_weighings(fine_weighing, n_subcells_per_axis):
    """Group fine SIMP weights by coarse macroelement and fine subcell."""

    n = _validate_n_subcells_per_axis(n_subcells_per_axis)
    fine_weighing = jnp.asarray(fine_weighing)
    n_dims = _validate_n_dims(fine_weighing.ndim)
    if any(size % n != 0 for size in fine_weighing.shape):
        raise ValueError(
            "Fine-grid element counts must be divisible by n_subcells_per_axis."
        )

    coarse_shape = tuple(size // n for size in fine_weighing.shape)
    interleaved_shape = tuple(
        entry
        for coarse_size in coarse_shape
        for entry in (coarse_size, n)
    )
    coarse_axes = tuple(range(0, 2 * n_dims, 2))
    subcell_axes = tuple(range(1, 2 * n_dims, 2))
    grouped = fine_weighing.reshape(interleaved_shape).transpose(
        coarse_axes + subcell_axes
    )
    return grouped.reshape((*coarse_shape, n**n_dims))


def _validate_galerkin_local_operators(galerkin_local_operators, n_dims):
    n_dims = _validate_n_dims(n_dims)
    galerkin_local_operators = jnp.asarray(galerkin_local_operators)
    n_dofs_element = n_dims * 2**n_dims
    if (
        galerkin_local_operators.ndim != 3
        or galerkin_local_operators.shape[1:]
        != (n_dofs_element, n_dofs_element)
    ):
        raise ValueError(
            "Galerkin local operators have an incompatible element shape."
        )
    n = _infer_n_subcells_per_axis(
        galerkin_local_operators.shape[0],
        n_dims,
    )
    return galerkin_local_operators, n


def _construct_level_elements2dofs(level, n_dofs_element):
    if n_dofs_element % level.n_nodes_element != 0:
        raise ValueError(
            "Element matrices are incompatible with the level topology."
        )
    n_dofs_node = n_dofs_element // level.n_nodes_element
    elements2nodes = construct_elements2nodes(level)
    elements2dofs = jnp.asarray(
        construct_elements2dofs(
            elements2nodes,
            n_nodes=level.n_nodes,
            n_dofs_node=n_dofs_node,
        )
    )
    return elements2dofs, n_dofs_node


def _galerkin_chunk_size(n_elements, n_subcell_operators):
    """Choose a bounded element chunk from the local-operator workload."""

    return min(
        n_elements,
        max(1, _GALERKIN_CHUNK_SUBCELL_BUDGET // n_subcell_operators),
    )


def _contract_galerkin_chunked(
    grouped,
    galerkin_local_operators,
    local_values,
    *,
    chunk_size,
):
    """Apply the implicit Galerkin contraction in bounded element chunks."""

    def apply_element(element_inputs):
        weighing, local_value = element_inputs
        return jnp.einsum(
            "s,sij,j->i",
            weighing,
            galerkin_local_operators,
            local_value,
        )

    return jax.lax.map(
        apply_element,
        (grouped, local_values),
        batch_size=chunk_size,
    )


def extract_structured_local_values(level, value):
    """Extract component-major Q1 element vectors without connectivity indices."""

    value = jnp.asarray(value)
    n_dims = _validate_n_dims(level.n_dims)
    expected_spatial_shape = tuple(map(int, level.nodes))
    if value.ndim != n_dims + 1 or value.shape[1:] != expected_spatial_shape:
        raise ValueError(
            "Expected a component-first nodal field with spatial shape "
            f"{expected_spatial_shape}, got {value.shape}."
        )

    spatial_axes = "HW" if n_dims == 2 else "DHW"
    patches = jax.lax.conv_general_dilated_patches(
        value[jnp.newaxis, ...],
        filter_shape=(2,) * n_dims,
        window_strides=(1,) * n_dims,
        padding="VALID",
        dimension_numbers=(
            f"NC{spatial_axes}",
            f"OI{spatial_axes}",
            f"NC{spatial_axes}",
        ),
    )[0]
    return einops.rearrange(
        patches,
        "local_dof ... -> (...) local_dof",
    )


def assemble_structured_local_results(level, local_results):
    """Sum component-major Q1 element vectors into a nodal field."""

    local_results = jnp.asarray(local_results)
    if local_results.ndim != 2 or local_results.shape[0] != level.n_elements:
        raise ValueError(
            "Expected one local result vector for every element on the level."
        )
    if local_results.shape[1] % level.n_nodes_element != 0:
        raise ValueError(
            "Local result vectors are incompatible with the level topology."
        )

    n_dims = _validate_n_dims(level.n_dims)
    element_axes = tuple(f"x{axis}" for axis in range(n_dims))
    corner_axes = tuple(f"x{axis}c" for axis in range(n_dims))
    element_group = " ".join(element_axes)
    corner_group = " ".join(corner_axes)
    interleaved_groups = " ".join(
        f"({element_axis} {corner_axis})"
        for element_axis, corner_axis in zip(element_axes, corner_axes)
    )
    local = einops.rearrange(
        local_results,
        f"({element_group}) (c {corner_group}) -> c {interleaved_groups}",
        **{
            **dict(zip(element_axes, map(int, level.elements))),
            **dict.fromkeys(corner_axes, 2),
        },
    )
    local = jax.lax.pad(
        local,
        jnp.asarray(0, dtype=local.dtype),
        ((0, 0, 0),) + ((1, 1, 0),) * n_dims,
    )
    return einops.reduce(
        local,
        f"c {interleaved_groups} -> c {element_group}",
        "sum",
        **dict.fromkeys(corner_axes, 2),
    )


def materialize_galerkin_element_matrices(
    fine_weighing,
    galerkin_local_operators,
):
    """Materialize heterogeneous macroelement matrices from fine weights."""

    fine_weighing = jnp.asarray(fine_weighing)
    galerkin_local_operators, n = _validate_galerkin_local_operators(
        galerkin_local_operators,
        fine_weighing.ndim,
    )
    grouped = group_fine_weighings(fine_weighing, n).reshape(
        -1,
        galerkin_local_operators.shape[0],
    )
    return jnp.einsum("es,sij->eij", grouped, galerkin_local_operators)


def coarsen_heterogeneous_element_matrices(level, element_matrices):
    """Galerkin-project heterogeneous element matrices to the next level.

    ``element_matrices`` contains one dense matrix for every element on
    ``level``. The returned matrices correspond to the dyadically coarsened
    level and are constructed recursively as

        K_parent = sum_s Q_s.T @ K_child_s @ Q_s.
    """

    if any(int(size) % 2 != 0 for size in level.elements):
        raise ValueError(
            "Recursive Galerkin materialization requires even element counts."
        )

    element_matrices = jnp.asarray(element_matrices)
    n_dofs_element = level.n_dims * level.n_nodes_element
    expected_shape = (
        level.n_elements,
        n_dofs_element,
        n_dofs_element,
    )
    if element_matrices.shape != expected_shape:
        raise ValueError(
            f"Expected element matrix shape {expected_shape}, "
            f"got {element_matrices.shape}."
        )

    n_dims = level.n_dims
    coarse_shape = tuple(int(size) // 2 for size in level.elements)
    interleaved_shape = tuple(
        entry
        for coarse_size in coarse_shape
        for entry in (coarse_size, 2)
    )
    matrix_field = element_matrices.reshape(
        (*map(int, level.elements), n_dofs_element, n_dofs_element)
    )
    coarse_axes = tuple(range(0, 2 * n_dims, 2))
    child_axes = tuple(range(1, 2 * n_dims, 2))
    grouped = matrix_field.reshape(
        (*interleaved_shape, n_dofs_element, n_dofs_element)
    ).transpose(
        coarse_axes
        + child_axes
        + (2 * n_dims, 2 * n_dims + 1)
    )
    grouped = grouped.reshape(
        -1,
        2**n_dims,
        n_dofs_element,
        n_dofs_element,
    )

    interpolation = construct_elastic_interpolation_matrices(
        2,
        n_dims=n_dims,
        dtype=element_matrices.dtype,
    )
    return jnp.einsum(
        "sia,esij,sjb->eab",
        interpolation,
        grouped,
        interpolation,
    )


def heterogeneous_matvec(
    level,
    value,
    element_matrices,
    restrictions=None,
):
    """Apply arbitrary per-element matrices using gather, multiply, scatter.

    As with Topax's existing matrix-free operator, constrained input entries
    are expected to have already been zeroed; restrictions masks output.
    """

    value = jnp.asarray(value)
    element_matrices = jnp.asarray(element_matrices)
    if (
        element_matrices.ndim != 3
        or element_matrices.shape[0] != level.n_elements
        or element_matrices.shape[1] != element_matrices.shape[2]
    ):
        raise ValueError(
            "Expected one square local matrix for every element on the level."
        )

    n_dofs_element = element_matrices.shape[1]
    elements2dofs, n_dofs_node = _construct_level_elements2dofs(
        level,
        n_dofs_element,
    )
    expected_shape = (n_dofs_node, *map(int, level.nodes))
    if value.shape != expected_shape:
        raise ValueError(f"Expected value shape {expected_shape}, got {value.shape}.")

    local_values = value.reshape(-1)[elements2dofs]
    local_results = jnp.einsum("eij,ej->ei", element_matrices, local_values)
    result = jnp.zeros(value.size, dtype=local_results.dtype)
    result = result.at[elements2dofs].add(local_results).reshape(value.shape)
    if restrictions is not None:
        result = result * restrictions
    return result


def heterogeneous_diagonal(level, element_matrices):
    """Assemble the diagonal of arbitrary per-element matrices."""

    element_matrices = jnp.asarray(element_matrices)
    if (
        element_matrices.ndim != 3
        or element_matrices.shape[0] != level.n_elements
        or element_matrices.shape[1] != element_matrices.shape[2]
    ):
        raise ValueError(
            "Expected one square local matrix for every element on the level."
        )

    n_dofs_element = element_matrices.shape[1]
    elements2dofs, n_dofs_node = _construct_level_elements2dofs(
        level,
        n_dofs_element,
    )
    local_diagonals = jnp.diagonal(element_matrices, axis1=1, axis2=2)
    diagonal = jnp.zeros(n_dofs_node * level.n_nodes, dtype=element_matrices.dtype)
    return diagonal.at[elements2dofs].add(local_diagonals).reshape(
        n_dofs_node,
        *map(int, level.nodes),
    )


def galerkin_matvec(
    level,
    value,
    fine_weighing,
    galerkin_local_operators,
    restrictions=None,
    *,
    slab_size=DEFAULT_GALERKIN_SLAB_SIZE,
):
    """Apply implicit Galerkin using slab-local intermediate arrays.

    Fine-weight grouping, structured nodal extraction, local contraction, and
    nodal assembly occur inside slabs along the first element axis, avoiding
    full-grid element-local input and output vectors.

    A final partial slab is represented by zero-weighted padding and cropped
    from the returned nodal field. Neighboring slabs both contribute to their
    shared nodal plane.
    """

    if isinstance(slab_size, bool) or not isinstance(slab_size, numbers.Integral):
        raise ValueError("slab_size must be a positive integer.")
    slab_size = int(slab_size)
    if slab_size < 1:
        raise ValueError("slab_size must be a positive integer.")

    operators, n = _validate_galerkin_local_operators(
        galerkin_local_operators,
        level.n_dims,
    )
    elements = tuple(map(int, level.elements))
    slab_size = min(slab_size, elements[0])
    n_dofs_element = operators.shape[1]
    if n_dofs_element % level.n_nodes_element != 0:
        raise ValueError(
            "Galerkin operators are incompatible with the level topology."
        )
    n_dofs_node = n_dofs_element // level.n_nodes_element

    expected_value_shape = (n_dofs_node, *map(int, level.nodes))
    value = jnp.asarray(value)
    if value.shape != expected_value_shape:
        raise ValueError(
            f"Expected value shape {expected_value_shape}, got {value.shape}."
        )

    expected_fine_shape = tuple(size * n for size in elements)
    fine_weighing = jnp.asarray(fine_weighing)
    if fine_weighing.shape != expected_fine_shape:
        raise ValueError(
            "Fine weighing and coarse level have incompatible shapes: "
            f"expected {expected_fine_shape}, got {fine_weighing.shape}."
        )

    n_slabs = (elements[0] + slab_size - 1) // slab_size
    padded_first_axis = n_slabs * slab_size
    padding = padded_first_axis - elements[0]
    if padding:
        value_padding = (
            ((0, 0), (0, padding))
            + ((0, 0),) * (level.n_dims - 1)
        )
        value = jnp.pad(value, value_padding)
        weighing_padding = (
            ((0, padding * n),)
            + ((0, 0),) * (level.n_dims - 1)
        )
        fine_weighing = jnp.pad(
            fine_weighing,
            weighing_padding,
        )

    tile_elements = np.array(
        [slab_size, *elements[1:]],
        dtype=int,
    )
    tile_sizes = level.element_sizes * tile_elements
    tile_level = Level(tile_sizes, tile_elements)
    chunk_size = _galerkin_chunk_size(
        tile_level.n_elements,
        operators.shape[0],
    )
    value_slice_shape = (
        n_dofs_node,
        slab_size + 1,
        *(size + 1 for size in elements[1:]),
    )
    fine_slice_shape = (
        slab_size * n,
        *(size * n for size in elements[1:]),
    )
    zero_spatial_start = (0,) * (level.n_dims - 1)

    def apply_slab(slab_index, result):
        coarse_start = slab_index * slab_size
        fine_start = coarse_start * n
        local_value = jax.lax.dynamic_slice(
            value,
            (0, coarse_start, *zero_spatial_start),
            value_slice_shape,
        )
        local_weighing = jax.lax.dynamic_slice(
            fine_weighing,
            (fine_start, *zero_spatial_start),
            fine_slice_shape,
        )
        grouped = group_fine_weighings(local_weighing, n).reshape(
            tile_level.n_elements,
            operators.shape[0],
        )
        local_values = extract_structured_local_values(tile_level, local_value)
        local_results = _contract_galerkin_chunked(
            grouped,
            operators,
            local_values,
            chunk_size=chunk_size,
        )
        local_result = assemble_structured_local_results(
            tile_level,
            local_results,
        )

        previous = jax.lax.dynamic_slice(
            result,
            (0, coarse_start, *zero_spatial_start),
            value_slice_shape,
        )
        return jax.lax.dynamic_update_slice(
            result,
            previous + local_result,
            (0, coarse_start, *zero_spatial_start),
        )

    result = jax.lax.fori_loop(
        0,
        n_slabs,
        apply_slab,
        jnp.zeros_like(value),
    )
    result = jax.lax.slice_in_dim(
        result,
        0,
        elements[0] + 1,
        axis=1,
    )
    if restrictions is not None:
        result = result * restrictions
    return result


def galerkin_diagonal(
    level,
    fine_weighing,
    galerkin_local_operators,
    *,
    slab_size=DEFAULT_GALERKIN_SLAB_SIZE,
):
    """Assemble the implicit Galerkin diagonal with slab-local arrays."""

    if isinstance(slab_size, bool) or not isinstance(slab_size, numbers.Integral):
        raise ValueError("slab_size must be a positive integer.")
    slab_size = int(slab_size)
    if slab_size < 1:
        raise ValueError("slab_size must be a positive integer.")

    operators, n = _validate_galerkin_local_operators(
        galerkin_local_operators,
        level.n_dims,
    )
    elements = tuple(map(int, level.elements))
    slab_size = min(slab_size, elements[0])
    n_dofs_element = operators.shape[1]
    if n_dofs_element % level.n_nodes_element != 0:
        raise ValueError(
            "Galerkin operators are incompatible with the level topology."
        )
    n_dofs_node = n_dofs_element // level.n_nodes_element

    expected_fine_shape = tuple(size * n for size in elements)
    fine_weighing = jnp.asarray(fine_weighing)
    if fine_weighing.shape != expected_fine_shape:
        raise ValueError(
            "Fine weighing and coarse level have incompatible shapes: "
            f"expected {expected_fine_shape}, got {fine_weighing.shape}."
        )

    n_slabs = (elements[0] + slab_size - 1) // slab_size
    padded_first_axis = n_slabs * slab_size
    padding = padded_first_axis - elements[0]
    if padding:
        weighing_padding = (
            ((0, padding * n),)
            + ((0, 0),) * (level.n_dims - 1)
        )
        fine_weighing = jnp.pad(
            fine_weighing,
            weighing_padding,
        )

    tile_elements = np.array(
        [slab_size, *elements[1:]],
        dtype=int,
    )
    tile_sizes = level.element_sizes * tile_elements
    tile_level = Level(tile_sizes, tile_elements)
    fine_slice_shape = (
        slab_size * n,
        *(size * n for size in elements[1:]),
    )
    result_shape = (
        n_dofs_node,
        padded_first_axis + 1,
        *(size + 1 for size in elements[1:]),
    )
    result_slice_shape = (
        n_dofs_node,
        slab_size + 1,
        *(size + 1 for size in elements[1:]),
    )
    operator_diagonals = jnp.diagonal(operators, axis1=1, axis2=2)
    zero_spatial_start = (0,) * (level.n_dims - 1)

    def apply_slab(slab_index, result):
        coarse_start = slab_index * slab_size
        fine_start = coarse_start * n
        local_weighing = jax.lax.dynamic_slice(
            fine_weighing,
            (fine_start, *zero_spatial_start),
            fine_slice_shape,
        )
        grouped = group_fine_weighings(local_weighing, n).reshape(
            tile_level.n_elements,
            operators.shape[0],
        )
        local_diagonals = jnp.einsum(
            "es,si->ei",
            grouped,
            operator_diagonals,
        )
        local_result = assemble_structured_local_results(
            tile_level,
            local_diagonals,
        )

        previous = jax.lax.dynamic_slice(
            result,
            (0, coarse_start, *zero_spatial_start),
            result_slice_shape,
        )
        return jax.lax.dynamic_update_slice(
            result,
            previous + local_result,
            (0, coarse_start, *zero_spatial_start),
        )

    result = jax.lax.fori_loop(
        0,
        n_slabs,
        apply_slab,
        jnp.zeros(result_shape, dtype=operators.dtype),
    )
    return jax.lax.slice_in_dim(
        result,
        0,
        elements[0] + 1,
        axis=1,
    )
