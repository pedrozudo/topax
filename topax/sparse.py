import dataclasses
from dataclasses import field

import numpy as np

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PhysicsCSRStructure:
    """Pure data required to assemble a lower-triangular physics CSR matrix."""

    csr_indices: jax.Array
    csr_indptr: jax.Array
    contribution_to_csr: jax.Array
    restricted_diagonal_positions: jax.Array
    n_dofs_element: int = field(metadata={"static": True})
    n_dofs_node: int = field(metadata={"static": True})
    n_dofs: int = field(metadata={"static": True})
    vector_shape: tuple[int, ...] = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HelmholtzCSRStructure:
    """Pure data for a fixed lower-triangular Helmholtz CSR matrix."""

    csr_indices: jax.Array
    csr_indptr: jax.Array
    csr_data: jax.Array
    n_dofs_node: int = field(metadata={"static": True})
    n_dofs: int = field(metadata={"static": True})
    vector_shape: tuple[int, ...] = field(metadata={"static": True})


def _assemble_physics_csr_values(structure, element_matrices):
    """Assemble element matrices into the fixed physics CSR pattern.

    Restricted DOFs are retained as decoupled diagonal entries. Their
    artificial diagonal uses the largest local diagonal contribution, which
    keeps the rule independent of whether the element matrices originated
    from a homogeneous operator or a heterogeneous Galerkin hierarchy.
    """

    element_matrices = jnp.asarray(element_matrices)
    expected_shape = (
        structure.contribution_to_csr.shape[0],
        structure.n_dofs_element,
        structure.n_dofs_element,
    )
    if element_matrices.shape != expected_shape:
        raise ValueError(
            f"Expected element matrix shape {expected_shape}, "
            f"got {element_matrices.shape}."
        )

    contributions = element_matrices.reshape(-1)
    mapping = structure.contribution_to_csr.reshape(-1)
    nse = structure.csr_indices.shape[0]
    values = jnp.bincount(
        mapping,
        weights=contributions,
        length=nse + 1,
    )[:-1]
    dirichlet_diagonal = jnp.max(
        jnp.diagonal(element_matrices, axis1=1, axis2=2)
    )
    return values.at[structure.restricted_diagonal_positions].set(
        dirichlet_diagonal
    )


def compute_physics_csr_values(structure, local_material_property, weighing):
    """Assemble homogeneous numeric values for a physics CSR structure."""

    local_material_property = jnp.asarray(local_material_property)
    dense_shape = (
        structure.n_dofs_element,
        structure.n_dofs_element,
    )
    n_dims = len(structure.vector_shape) - 1
    operator_shape = (
        structure.n_dofs_element,
        structure.n_dofs_node,
        *((2,) * n_dims),
    )
    if local_material_property.shape not in (dense_shape, operator_shape):
        raise ValueError(
            f"Expected local material property shape {dense_shape} or "
            f"{operator_shape}, got {local_material_property.shape}."
        )

    weighing = jnp.asarray(weighing)
    n_elements = structure.contribution_to_csr.shape[0]
    if weighing.size != n_elements:
        raise ValueError(
            f"Expected {n_elements} element weights, got {weighing.size}."
        )

    local_matrix = local_material_property.reshape(dense_shape)
    element_matrices = (
        weighing.reshape(-1, 1, 1) * local_matrix.reshape(1, *dense_shape)
    )
    return _assemble_physics_csr_values(structure, element_matrices)


def compute_heterogeneous_physics_csr_values(structure, element_matrices):
    """Assemble CSR values from one dense local matrix per element."""

    return _assemble_physics_csr_values(structure, element_matrices)


def unpack_sparse_vector(structure, vector):
    return vector.reshape(structure.vector_shape)


def sparse_todense(structure, values):
    """Materialize the symmetric matrix represented by lower-triangle CSR."""

    import jax.experimental.sparse as jsparse

    lower = jsparse.BCSR(
        (values, structure.csr_indices, structure.csr_indptr),
        shape=(structure.n_dofs, structure.n_dofs),
    ).todense()
    return lower + lower.T - jnp.diag(jnp.diag(lower))


def construct_sparse_physics_structure(
    level,
    *,
    n_dofs_node,
    restriction,
):
    if isinstance(n_dofs_node, (bool, np.bool_)) or not isinstance(
        n_dofs_node, (int, np.integer)
    ):
        raise ValueError("n_dofs_node must be a positive integer.")
    if n_dofs_node < 1:
        raise ValueError("n_dofs_node must be a positive integer.")

    n_dofs_node = int(n_dofs_node)
    n_dofs_element = n_dofs_node * level.n_nodes_element
    n_dofs = n_dofs_node * level.n_nodes

    elements2nodes = construct_elements2nodes(level)

    elements2dofs = construct_elements2dofs(
        elements2nodes,
        n_nodes=level.n_nodes,
        n_dofs_node=n_dofs_node,
    )

    if elements2dofs.shape != (
        level.n_elements,
        n_dofs_element,
    ):
        raise ValueError(
            "Element-to-DOF connectivity does not match n_dofs_node."
        )

    restriction = np.asarray(
        jax.device_get(restriction),
        dtype=bool,
    ).reshape(-1)

    if restriction.size != n_dofs:
        raise ValueError("Restriction size does not match the number of DOFs.")

    (
        csr_indices,
        csr_indptr,
        contribution_to_csr,
        restricted_diagonal_positions,
    ) = construct_csr_mapping(
        elements2dofs,
        n_dofs=n_dofs,
        restriction=restriction,
    )

    return PhysicsCSRStructure(
        csr_indices=jnp.asarray(csr_indices),
        csr_indptr=jnp.asarray(csr_indptr),
        contribution_to_csr=jnp.asarray(contribution_to_csr),
        restricted_diagonal_positions=jnp.asarray(restricted_diagonal_positions),
        n_dofs_element=n_dofs_element,
        n_dofs_node=n_dofs_node,
        n_dofs=n_dofs,
        vector_shape=(n_dofs_node, *map(int, level.nodes)),
    )


def construct_sparse_helmholtz_structure(level, local_operator):
    n_dofs_element = local_operator.shape[0]

    if n_dofs_element != level.n_nodes_element:
        raise ValueError("Helmholtz requires one DOF per node.")

    n_dofs_node = 1
    n_dofs = level.n_nodes

    local_operator = np.asarray(jax.device_get(local_operator))

    if local_operator.size != (n_dofs_element * n_dofs_element):
        raise ValueError(
            "Local Helmholtz operator must represent a square local operator."
        )

    elements2nodes = construct_elements2nodes(level)

    elements2dofs = construct_elements2dofs(
        elements2nodes, n_nodes=level.n_nodes, n_dofs_node=n_dofs_node
    )

    (csr_indices, csr_indptr, contribution_to_csr, _) = construct_csr_mapping(
        elements2dofs, n_dofs=n_dofs
    )

    element_matrix = local_operator.reshape(-1)

    contributions = np.broadcast_to(
        element_matrix,
        (level.n_elements, element_matrix.size),
    ).reshape(-1)

    csr_data = np.bincount(
        contribution_to_csr.reshape(-1),
        weights=contributions,
        minlength=csr_indices.size + 1,
    )[:-1]

    return HelmholtzCSRStructure(
        csr_indices=jnp.asarray(csr_indices),
        csr_indptr=jnp.asarray(csr_indptr),
        csr_data=jnp.asarray(csr_data),
        n_dofs_node=n_dofs_node,
        n_dofs=n_dofs,
        vector_shape=(n_dofs_node, *map(int, level.nodes)),
    )


def construct_csr_mapping(
    elements2dofs,
    n_dofs,
    restriction=None,
):
    """
    Construct the CSR sparsity pattern and map every local element-matrix
    contribution to its position in the CSR data array.

    If restrictions are present, contributions touching restricted DOFs
    are mapped to the sentinel value `nse`. Restricted DOFs are retained
    in the matrix as diagonal-only entries.
    """
    elements2dofs = np.asarray(
        elements2dofs,
        dtype=np.int64,
    )

    n_elements, n_dofs_element = elements2dofs.shape

    if restriction is None:
        restriction = np.ones(
            n_dofs,
            dtype=bool,
        )
    else:
        restriction = np.asarray(
            restriction,
            dtype=bool,
        ).reshape(-1)

        if restriction.size != n_dofs:
            raise ValueError("Restriction size does not match n_dofs.")

    # Encode (row, col) as row * n_dofs + col.
    # Sorting these encoded positions is equivalent to CSR row-major order.
    row_col = (elements2dofs[:, :, None] * n_dofs + elements2dofs[:, None, :]).reshape(
        -1
    )

    element_dof_active = restriction[elements2dofs]

    contribution_active = (
        element_dof_active[:, :, None] & element_dof_active[:, None, :]
    ).reshape(-1)

    # Xolky/cuDSS consumes the lower triangle of the symmetric matrix.
    rows_all = np.broadcast_to(
        elements2dofs[:, :, None],
        (n_elements, n_dofs_element, n_dofs_element),
    ).reshape(-1)
    cols_all = np.broadcast_to(
        elements2dofs[:, None, :],
        (n_elements, n_dofs_element, n_dofs_element),
    ).reshape(-1)
    contribution_active &= rows_all >= cols_all

    active_row_col = row_col[contribution_active]

    restricted_dofs = np.flatnonzero(~restriction).astype(np.int64)

    restricted_diagonal_row_col = restricted_dofs * n_dofs + restricted_dofs

    all_row_col = np.concatenate(
        (
            active_row_col,
            restricted_diagonal_row_col,
        )
    )

    unique_row_col, inverse = np.unique(
        all_row_col,
        return_inverse=True,
    )

    nse = unique_row_col.size
    n_active_contributions = active_row_col.size

    # Local contributions touching restricted DOFs get nse, which is one
    # position past the final valid CSR entry and acts as a sentinel bin.
    contribution_to_csr = np.full(
        row_col.size,
        nse,
        dtype=np.int32,
    )

    contribution_to_csr[contribution_active] = inverse[:n_active_contributions].astype(
        np.int32
    )

    contribution_to_csr = contribution_to_csr.reshape(
        n_elements,
        n_dofs_element * n_dofs_element,
    )

    restricted_diagonal_positions = inverse[n_active_contributions:].astype(np.int32)

    rows = unique_row_col // n_dofs

    csr_indices = (unique_row_col % n_dofs).astype(np.int32)

    entries_per_row = np.bincount(
        rows,
        minlength=n_dofs,
    )

    csr_indptr = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(
                entries_per_row,
                dtype=np.int64,
            ),
        )
    ).astype(np.int32)

    return (
        csr_indices,
        csr_indptr,
        contribution_to_csr,
        restricted_diagonal_positions,
    )


def construct_elements2nodes(level):
    if level.n_dims == 2:
        return elements2nodes_2D(*level.elements)

    if level.n_dims == 3:
        return elements2nodes_3D(*level.elements)

    raise NotImplementedError(f"Unsupported dimension: {level.n_dims}")


def elements2nodes_2D(
    nex,
    ney,
):
    x = np.arange(
        nex,
        dtype=np.int64,
    ) * (ney + 1)

    y = np.arange(
        ney,
        dtype=np.int64,
    )

    elements2nodes = x[:, None] + y

    offsets = np.array(
        [
            0,
            1,
            ney + 1,
            ney + 2,
        ],
        dtype=np.int64,
    )

    elements2nodes = elements2nodes[..., None] + offsets

    return elements2nodes.reshape(
        -1,
        4,
    )


def elements2nodes_3D(
    nex,
    ney,
    nez,
):
    nodes_y = ney + 1
    nodes_z = nez + 1

    stride_x = nodes_y * nodes_z
    stride_y = nodes_z

    x = (
        np.arange(
            nex,
            dtype=np.int64,
        )
        * stride_x
    )

    y = (
        np.arange(
            ney,
            dtype=np.int64,
        )
        * stride_y
    )

    z = np.arange(
        nez,
        dtype=np.int64,
    )

    elements2nodes = x[:, None, None] + y[None, :, None] + z[None, None, :]

    offsets = np.array(
        [
            0,
            1,
            stride_y,
            stride_y + 1,
            stride_x,
            stride_x + 1,
            stride_x + stride_y,
            stride_x + stride_y + 1,
        ],
        dtype=np.int64,
    )

    elements2nodes = elements2nodes[..., None] + offsets

    return elements2nodes.reshape(
        -1,
        8,
    )


def construct_elements2dofs(
    elements2nodes,
    n_nodes,
    n_dofs_node,
):
    dof_offsets = (
        np.arange(
            n_dofs_node,
            dtype=np.int64,
        )
        * n_nodes
    )

    elements2dofs = elements2nodes[:, None, :] + dof_offsets[None, :, None]

    return elements2dofs.reshape(
        elements2nodes.shape[0],
        -1,
    )
