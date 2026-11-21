import numpy as np

import jax.numpy as jnp
import pytest

from topax.grid import Level
from topax.sparse import (
    compute_heterogeneous_physics_csr_values,
    compute_physics_csr_values,
    construct_elements2dofs,
    construct_elements2nodes,
    construct_sparse_physics_structure,
    elements2nodes_2D,
    elements2nodes_3D,
)


def test_single_quadrilateral_node_ordering():
    actual = elements2nodes_2D(
        nex=1,
        ney=1,
    )

    expected = np.array(
        [
            [0, 1, 2, 3],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(actual, expected)


def test_single_hexahedron_node_ordering():
    actual = elements2nodes_3D(
        nex=1,
        ney=1,
        nez=1,
    )

    expected = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(actual, expected)


def test_rectangular_hexahedral_mesh_connectivity():
    actual = elements2nodes_3D(
        nex=2,
        ney=1,
        nez=2,
    )

    expected = np.array(
        [
            [0, 1, 3, 4, 6, 7, 9, 10],
            [1, 2, 4, 5, 7, 8, 10, 11],
            [6, 7, 9, 10, 12, 13, 15, 16],
            [7, 8, 10, 11, 13, 14, 16, 17],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(actual, expected)


def test_construct_elements2nodes_dispatches_to_3d():
    level = Level(
        sizes=[2.0, 1.0, 3.0],
        elements=[2, 1, 2],
    )

    actual = construct_elements2nodes(level)
    expected = elements2nodes_3D(2, 1, 2)

    assert actual.shape == (level.n_elements, level.n_nodes_element)
    np.testing.assert_array_equal(actual, expected)


def test_three_component_dof_ordering_for_single_hexahedron():
    elements2nodes = elements2nodes_3D(
        nex=1,
        ney=1,
        nez=1,
    )

    actual = construct_elements2dofs(
        elements2nodes,
        n_nodes=8,
        n_dofs_node=3,
    )

    expected = np.array(
        [
            np.arange(24),
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(actual, expected)


def test_physics_csr_structure_is_independent_of_local_operator():
    level = Level(sizes=[2.0, 1.0], elements=[2, 1])
    restriction = np.ones((2, *level.nodes), dtype=bool)

    structure = construct_sparse_physics_structure(
        level,
        n_dofs_node=2,
        restriction=restriction,
    )

    assert not hasattr(structure, "local_material_property")
    assert structure.n_dofs_node == 2
    assert structure.n_dofs_element == 2 * level.n_nodes_element
    assert structure.n_dofs == 2 * level.n_nodes
    assert structure.vector_shape == (2, *map(int, level.nodes))


@pytest.mark.parametrize("n_dofs_node", [True, 0, -1, 1.5])
def test_physics_csr_structure_rejects_invalid_dofs_per_node(n_dofs_node):
    level = Level(sizes=[1.0, 1.0], elements=[1, 1])
    restriction = np.ones((2, *level.nodes), dtype=bool)

    with pytest.raises(ValueError, match="positive integer"):
        construct_sparse_physics_structure(
            level,
            n_dofs_node=n_dofs_node,
            restriction=restriction,
        )


def test_physics_csr_numeric_assembly_validates_numeric_inputs():
    level = Level(sizes=[1.0, 1.0], elements=[1, 1])
    restriction = np.ones((2, *level.nodes), dtype=bool)
    structure = construct_sparse_physics_structure(
        level,
        n_dofs_node=2,
        restriction=restriction,
    )
    n_local = structure.n_dofs_element

    with pytest.raises(ValueError, match="local material property shape"):
        compute_physics_csr_values(
            structure,
            jnp.eye(n_local - 1),
            jnp.ones(level.n_elements),
        )

    with pytest.raises(ValueError, match="local material property shape"):
        compute_physics_csr_values(
            structure,
            jnp.ones(n_local**2),
            jnp.ones(level.n_elements),
        )

    with pytest.raises(ValueError, match="element weights"):
        compute_physics_csr_values(
            structure,
            jnp.eye(n_local),
            jnp.ones(level.n_elements + 1),
        )


def test_homogeneous_and_heterogeneous_assembly_use_the_same_kernel():
    level = Level(sizes=[2.0, 1.0], elements=[2, 1])
    restriction = np.ones((2, *level.nodes), dtype=bool)
    restriction[:, 0, 0] = False
    structure = construct_sparse_physics_structure(
        level,
        n_dofs_node=2,
        restriction=restriction,
    )
    n_local = structure.n_dofs_element
    local_matrix = jnp.diag(jnp.arange(1, n_local + 1, dtype=jnp.float64))
    weighing = jnp.asarray([0.5, 2.0], dtype=jnp.float64)
    element_matrices = weighing[:, None, None] * local_matrix[None, :, :]

    homogeneous = compute_physics_csr_values(
        structure,
        local_matrix,
        weighing,
    )
    heterogeneous = compute_heterogeneous_physics_csr_values(
        structure,
        element_matrices,
    )

    np.testing.assert_allclose(homogeneous, heterogeneous)
    expected_dirichlet_diagonal = jnp.max(
        jnp.diagonal(element_matrices, axis1=1, axis2=2)
    )
    np.testing.assert_allclose(
        homogeneous[structure.restricted_diagonal_positions],
        expected_dirichlet_diagonal,
    )
