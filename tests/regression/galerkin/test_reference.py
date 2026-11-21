import jax
import jax.numpy as jnp
import numpy as np
import pytest

import topax
import topax.galerkin as galerkin
from topax.sparse import construct_elements2dofs, construct_elements2nodes


jax.config.update("jax_enable_x64", True)


def test_chunked_galerkin_contraction_handles_remainder_under_jit():
    n_elements = 7
    n_subcells = 4
    n_local_dofs = 8
    grouped = jnp.sin(
        jnp.arange(n_elements * n_subcells, dtype=jnp.float64) / 7.0
    ).reshape(n_elements, n_subcells)
    local_operators = jnp.cos(
        jnp.arange(
            n_subcells * n_local_dofs**2,
            dtype=jnp.float64,
        )
        / 11.0
    ).reshape(n_subcells, n_local_dofs, n_local_dofs)
    local_values = jnp.sin(
        jnp.arange(n_elements * n_local_dofs, dtype=jnp.float64) / 13.0
    ).reshape(n_elements, n_local_dofs)

    actual = jax.jit(
        lambda weighings, operators, values: galerkin._contract_galerkin_chunked(
            weighings,
            operators,
            values,
            chunk_size=3,
        )
    )(grouped, local_operators, local_values)
    expected = jnp.einsum(
        "es,sij,ej->ei",
        grouped,
        local_operators,
        local_values,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_chunked_galerkin_contraction_vjps_match_whole_domain():
    grouped = jnp.arange(20, dtype=jnp.float64).reshape(5, 4) / 17.0
    local_operators = jnp.sin(jnp.arange(4 * 8 * 8, dtype=jnp.float64) / 19.0).reshape(
        4, 8, 8
    )
    local_values = jnp.cos(jnp.arange(5 * 8, dtype=jnp.float64) / 23.0).reshape(5, 8)

    def whole_domain(weighings, operators, values):
        result = jnp.einsum("es,sij,ej->ei", weighings, operators, values)
        return jnp.sum(result**2)

    def chunked(weighings, operators, values):
        result = galerkin._contract_galerkin_chunked(
            weighings,
            operators,
            values,
            chunk_size=3,
        )
        return jnp.sum(result**2)

    expected = jax.jit(jax.grad(whole_domain, argnums=(0, 1, 2)))(
        grouped,
        local_operators,
        local_values,
    )
    actual = jax.jit(jax.grad(chunked, argnums=(0, 1, 2)))(
        grouped,
        local_operators,
        local_values,
    )
    for actual_gradient, expected_gradient in zip(actual, expected):
        np.testing.assert_allclose(
            actual_gradient,
            expected_gradient,
            rtol=1e-13,
            atol=1e-13,
        )


def test_galerkin_chunk_size_bounds_subcell_work_and_small_problems():
    assert galerkin._galerkin_chunk_size(1_000_000, 8) == 65_536
    assert galerkin._galerkin_chunk_size(1_000_000, 64) == 8_192
    assert galerkin._galerkin_chunk_size(17, 8) == 17


def _construct_problem(elements=(4, 2, 2), with_restrictions=False, n_levels=2):
    elements = np.asarray(elements, dtype=int)
    grid = topax.construct_grid(
        n_levels=n_levels,
        sizes=elements.astype(np.float64),
        elements=elements,
    )
    elastic = topax.construct_elastic(
        elastic_module=1.0,
        poisson_ratio=0.3,
        thickness=1.0,
    )
    local_stiffnesses = tuple(
        topax.construct_local_stiffness(level, elastic) for level in grid.levels
    )

    restrictions = None
    if with_restrictions:
        fixed_face = {
            "center": [0.0, 0.5, 0.5],
            "size": [0.0, 1.0, 1.0],
        }
        restrictions = topax.construct_restrictions(
            grid,
            elastic,
            [[fixed_face], [fixed_face], [fixed_face]],
        )

    return grid, local_stiffnesses, restrictions


@pytest.mark.parametrize(
    ("elements", "n_components"),
    [
        ((3, 2), 2),
        ((2, 3, 1), 3),
    ],
)
def test_structured_local_assembly_matches_scatter(elements, n_components):
    grid = topax.construct_grid(
        n_levels=1,
        sizes=np.ones(len(elements)),
        elements=np.asarray(elements),
    )
    level = grid.fine_level
    n_local_dofs = n_components * level.n_nodes_element
    local_results = jnp.sin(
        jnp.arange(level.n_elements * n_local_dofs, dtype=jnp.float64)
    ).reshape(level.n_elements, n_local_dofs)

    elements2dofs = construct_elements2dofs(
        construct_elements2nodes(level),
        n_nodes=level.n_nodes,
        n_dofs_node=n_components,
    )
    expected = jnp.zeros(n_components * level.n_nodes, dtype=jnp.float64)
    expected = (
        expected.at[elements2dofs]
        .add(local_results)
        .reshape(
            n_components,
            *map(int, level.nodes),
        )
    )
    actual = jax.jit(
        galerkin.assemble_structured_local_results,
        static_argnums=0,
    )(level, local_results)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    value = jnp.cos(
        jnp.arange(n_components * level.n_nodes, dtype=jnp.float64)
    ).reshape(n_components, *map(int, level.nodes))
    expected_local_values = value.reshape(-1)[elements2dofs]
    actual_local_values = jax.jit(
        galerkin.extract_structured_local_values,
        static_argnums=0,
    )(level, value)
    np.testing.assert_allclose(
        actual_local_values, expected_local_values, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("with_restrictions", [False, True])
def test_restriction_is_adjoint_of_prolongation(with_restrictions):
    grid, _, restrictions = _construct_problem(
        elements=(4, 2, 2),
        with_restrictions=with_restrictions,
    )
    coarse_shape = (3, *map(int, grid.levels[1].nodes))
    fine_shape = (3, *map(int, grid.levels[0].nodes))
    coarse_value = jnp.sin(
        jnp.arange(np.prod(coarse_shape), dtype=jnp.float64) / 7.0
    ).reshape(coarse_shape)
    fine_value = jnp.cos(
        jnp.arange(np.prod(fine_shape), dtype=jnp.float64) / 11.0
    ).reshape(fine_shape)

    prolonged = topax.vcycle.prolongate_correction(
        coarse_value,
        grid,
        restrictions,
        level_index=0,
    )
    restricted = topax.vcycle.restrict_residual(
        fine_value,
        grid,
        restrictions,
        level_index=0,
    )

    np.testing.assert_allclose(
        jnp.vdot(prolonged, fine_value),
        jnp.vdot(coarse_value, restricted),
        rtol=1e-14,
        atol=1e-14,
    )


def test_local_interpolation_matches_topax_prolongation():
    grid, _, _ = _construct_problem(elements=(2, 2, 2))
    coarse_shape = (3, *map(int, grid.levels[1].nodes))
    coarse_value = jnp.arange(
        np.prod(coarse_shape),
        dtype=jnp.float64,
    ).reshape(coarse_shape)
    fine_value = topax.vcycle.prolongate_correction(
        coarse_value,
        grid,
        restrictions=None,
        level_index=0,
    )

    elements2nodes = construct_elements2nodes(grid.levels[0])
    elements2dofs = construct_elements2dofs(
        elements2nodes,
        n_nodes=grid.levels[0].n_nodes,
        n_dofs_node=3,
    )
    actual_local_values = fine_value.reshape(-1)[elements2dofs]

    interpolation = galerkin.construct_elastic_interpolation_matrices(
        n_subcells_per_axis=2,
        dtype=jnp.float64,
    )
    expected_local_values = jnp.einsum(
        "sij,j->si",
        interpolation,
        coarse_value.reshape(-1),
    )

    np.testing.assert_allclose(
        actual_local_values,
        expected_local_values,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("n_subcells_per_axis", [2, 4])
def test_galerkin_local_operators_are_symmetric_positive_semidefinite(
    n_subcells_per_axis,
):
    _, local_stiffnesses, _ = _construct_problem(
        elements=(n_subcells_per_axis,) * 3,
        n_levels=int(np.log2(n_subcells_per_axis)) + 1,
    )
    local_operators = galerkin.construct_galerkin_local_operators(
        local_stiffnesses[0],
        n_subcells_per_axis,
    )

    np.testing.assert_allclose(
        local_operators,
        jnp.swapaxes(local_operators, 1, 2),
        rtol=1e-14,
        atol=1e-14,
    )
    eigenvalues = jnp.linalg.eigvalsh(local_operators)
    assert float(jnp.min(eigenvalues)) >= -1e-12

    np.testing.assert_allclose(
        jnp.sum(local_operators, axis=0),
        local_stiffnesses[-1].reshape(24, 24),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("with_restrictions", [False, True])
def test_galerkin_matvec_and_diagonal_match_literal_projection(
    with_restrictions,
):
    grid, local_stiffnesses, restrictions = _construct_problem(
        elements=(4, 2, 2),
        with_restrictions=with_restrictions,
    )
    fine_restriction = None if restrictions is None else restrictions[0]
    coarse_restriction = None if restrictions is None else restrictions[1]
    fine_weighing = jnp.linspace(
        0.15,
        1.0,
        grid.levels[0].n_elements,
        dtype=jnp.float64,
    ).reshape(tuple(grid.levels[0].elements))
    coarse_shape = (3, *map(int, grid.levels[1].nodes))
    coarse_value = jnp.sin(
        jnp.arange(np.prod(coarse_shape), dtype=jnp.float64) / 5.0
    ).reshape(coarse_shape)
    if coarse_restriction is not None:
        coarse_value = coarse_value * coarse_restriction

    def literal_operator(value):
        fine_value = topax.vcycle.prolongate_correction(
            value,
            grid,
            restrictions,
            level_index=0,
        )
        fine_result = topax.operator.matrix_free_matvec(
            grid.levels[0],
            fine_value,
            local_stiffnesses[0],
            fine_weighing,
            fine_restriction,
        )
        return topax.vcycle.restrict_residual(
            fine_result,
            grid,
            restrictions,
            level_index=0,
        )

    local_operators = galerkin.construct_galerkin_local_operators(
        local_stiffnesses[0],
        n_subcells_per_axis=2,
    )
    implicit_operator = jax.jit(
        lambda value: galerkin.galerkin_matvec(
            grid.levels[1],
            value,
            fine_weighing,
            local_operators,
            coarse_restriction,
        )
    )
    actual = implicit_operator(coarse_value)
    expected = literal_operator(coarse_value)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    n_coarse_dofs = int(np.prod(coarse_shape))
    basis = jnp.eye(n_coarse_dofs, dtype=jnp.float64).reshape(
        n_coarse_dofs,
        *coarse_shape,
    )
    literal_columns = jax.vmap(literal_operator)(basis).reshape(
        n_coarse_dofs,
        n_coarse_dofs,
    )
    expected_diagonal = jnp.diag(literal_columns).reshape(coarse_shape)
    actual_diagonal = galerkin.galerkin_diagonal(
        grid.levels[1],
        fine_weighing,
        local_operators,
    )

    active = jnp.ones(coarse_shape, dtype=bool)
    if coarse_restriction is not None:
        active = coarse_restriction
    np.testing.assert_allclose(
        actual_diagonal[active],
        expected_diagonal[active],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("with_restrictions", [False, True])
def test_galerkin_matvec_matches_materialized_with_partial_final_slab(
    with_restrictions,
):
    grid, local_stiffnesses, restrictions = _construct_problem(
        elements=(10, 8, 8),
        with_restrictions=with_restrictions,
    )
    level = grid.levels[1]
    restriction = None if restrictions is None else restrictions[1]
    fine_weighing = jnp.linspace(
        0.15,
        1.0,
        grid.fine_level.n_elements,
        dtype=jnp.float64,
    ).reshape(tuple(grid.fine_level.elements))
    value_shape = (3, *map(int, level.nodes))
    value = jnp.sin(
        jnp.arange(np.prod(value_shape), dtype=jnp.float64) / 11.0
    ).reshape(value_shape)
    local_operators = galerkin.construct_galerkin_local_operators(
        local_stiffnesses[0],
        n_subcells_per_axis=2,
    )

    implicit = jax.jit(
        lambda value_, weighing_: galerkin.galerkin_matvec(
            level,
            value_,
            weighing_,
            local_operators,
            restriction,
            slab_size=4,
        )
    )
    materialized = jax.jit(
        lambda value_, weighing_: galerkin.heterogeneous_matvec(
            level,
            value_,
            galerkin.materialize_galerkin_element_matrices(
                weighing_,
                local_operators,
            ),
            restriction,
        )
    )

    expected = materialized(value, fine_weighing)
    actual = implicit(value, fine_weighing)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    cotangent = jnp.cos(
        jnp.arange(np.prod(value_shape), dtype=jnp.float64) / 13.0
    ).reshape(value_shape)

    def scalar_result(operator, value_, weighing_):
        return jnp.vdot(operator(value_, weighing_), cotangent).real

    expected_gradients = jax.grad(
        lambda value_, weighing_: scalar_result(
            materialized,
            value_,
            weighing_,
        ),
        argnums=(0, 1),
    )(value, fine_weighing)
    actual_gradients = jax.grad(
        lambda value_, weighing_: scalar_result(
            implicit,
            value_,
            weighing_,
        ),
        argnums=(0, 1),
    )(value, fine_weighing)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
    ):
        np.testing.assert_allclose(
            actual_gradient,
            expected_gradient,
            rtol=1e-12,
            atol=1e-12,
        )


def test_galerkin_diagonal_matches_materialized_with_partial_final_slab():
    grid, local_stiffnesses, _ = _construct_problem(elements=(10, 8, 8))
    level = grid.levels[1]
    fine_weighing = jnp.linspace(
        0.15,
        1.0,
        grid.fine_level.n_elements,
        dtype=jnp.float64,
    ).reshape(tuple(grid.fine_level.elements))
    local_operators = galerkin.construct_galerkin_local_operators(
        local_stiffnesses[0],
        n_subcells_per_axis=2,
    )

    implicit = jax.jit(
        lambda weighing: galerkin.galerkin_diagonal(
            level,
            weighing,
            local_operators,
            slab_size=4,
        )
    )
    materialized = jax.jit(
        lambda weighing: galerkin.heterogeneous_diagonal(
            level,
            galerkin.materialize_galerkin_element_matrices(
                weighing,
                local_operators,
            ),
        )
    )

    expected = materialized(fine_weighing)
    actual = implicit(fine_weighing)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    cotangent = jnp.cos(
        jnp.arange(actual.size, dtype=jnp.float64) / 17.0
    ).reshape(actual.shape)
    expected_gradient = jax.grad(
        lambda weighing: jnp.vdot(materialized(weighing), cotangent).real
    )(fine_weighing)
    actual_gradient = jax.grad(
        lambda weighing: jnp.vdot(implicit(weighing), cotangent).real
    )(fine_weighing)
    np.testing.assert_allclose(
        actual_gradient,
        expected_gradient,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("n_subcells_per_axis", [0, -1, 1.5, True])
def test_interpolation_rejects_invalid_subcell_counts(n_subcells_per_axis):
    with pytest.raises(ValueError, match="positive integer"):
        galerkin.construct_nodal_interpolation_matrices(n_subcells_per_axis)


def test_implicit_galerkin_matches_materialized_reference():
    grid, local_stiffnesses, _ = _construct_problem(elements=(4, 2, 2))
    fine_weighing = jnp.linspace(
        0.2,
        0.9,
        grid.levels[0].n_elements,
        dtype=jnp.float64,
    ).reshape(tuple(grid.levels[0].elements))
    local_operators = galerkin.construct_galerkin_local_operators(
        local_stiffnesses[0],
        n_subcells_per_axis=2,
    )
    element_matrices = galerkin.materialize_galerkin_element_matrices(
        fine_weighing,
        local_operators,
    )
    coarse_shape = (3, *map(int, grid.levels[1].nodes))
    value = jnp.cos(
        jnp.arange(np.prod(coarse_shape), dtype=jnp.float64) / 13.0
    ).reshape(coarse_shape)

    implicit_result = galerkin.galerkin_matvec(
        grid.levels[1],
        value,
        fine_weighing,
        local_operators,
    )
    materialized_result = galerkin.heterogeneous_matvec(
        grid.levels[1],
        value,
        element_matrices,
    )
    np.testing.assert_allclose(
        implicit_result,
        materialized_result,
        rtol=1e-12,
        atol=1e-12,
    )

    implicit_diagonal = galerkin.galerkin_diagonal(
        grid.levels[1],
        fine_weighing,
        local_operators,
    )
    materialized_diagonal = galerkin.heterogeneous_diagonal(
        grid.levels[1],
        element_matrices,
    )
    np.testing.assert_allclose(
        implicit_diagonal,
        materialized_diagonal,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("with_restrictions", [False, True])
def test_level_two_galerkin_matches_literal_projection(n_dims, with_restrictions):
    elements = np.full(n_dims, 4, dtype=int)
    grid = topax.construct_grid(
        n_levels=3,
        sizes=elements.astype(np.float64),
        elements=elements,
    )
    elastic = topax.construct_elastic(
        elastic_module=1.0,
        poisson_ratio=0.3,
        thickness=1.0,
    )
    local_stiffness = topax.construct_local_stiffness(grid.fine_level, elastic)

    restrictions = None
    if with_restrictions:
        fixed_face = {
            "center": [0.0] + [0.5] * (n_dims - 1),
            "size": [0.0] + [1.0] * (n_dims - 1),
        }
        restrictions = topax.construct_restrictions(
            grid,
            elastic,
            [[fixed_face] for _ in range(n_dims)],
        )

    fine_weighing = jnp.linspace(
        0.15,
        1.0,
        grid.fine_level.n_elements,
        dtype=jnp.float64,
    ).reshape(tuple(grid.fine_level.elements))
    coarse_shape = (n_dims, *map(int, grid.levels[2].nodes))
    coarse_value = jnp.sin(
        jnp.arange(np.prod(coarse_shape), dtype=jnp.float64) / 5.0
    ).reshape(coarse_shape)
    if restrictions is not None:
        coarse_value = coarse_value * restrictions[2]

    def literal_operator(value):
        for level_index in (1, 0):
            value = topax.vcycle.prolongate_correction(
                value,
                grid,
                restrictions,
                level_index,
            )
        value = topax.operator.matrix_free_matvec(
            grid.fine_level,
            value,
            local_stiffness,
            fine_weighing,
            None if restrictions is None else restrictions[0],
        )
        for level_index in (0, 1):
            value = topax.vcycle.restrict_residual(
                value,
                grid,
                restrictions,
                level_index,
            )
        return value

    local_operators = galerkin.construct_galerkin_local_operators(
        local_stiffness,
        n_subcells_per_axis=4,
        n_dims=n_dims,
    )
    actual = galerkin.galerkin_matvec(
        grid.levels[2],
        coarse_value,
        fine_weighing,
        local_operators,
        None if restrictions is None else restrictions[2],
    )
    expected = literal_operator(coarse_value)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    n_coarse_dofs = int(np.prod(coarse_shape))
    basis = jnp.eye(n_coarse_dofs, dtype=jnp.float64).reshape(
        n_coarse_dofs,
        *coarse_shape,
    )
    literal_columns = jax.vmap(literal_operator)(basis).reshape(
        n_coarse_dofs,
        n_coarse_dofs,
    )
    expected_diagonal = jnp.diag(literal_columns).reshape(coarse_shape)
    actual_diagonal = galerkin.galerkin_diagonal(
        grid.levels[2],
        fine_weighing,
        local_operators,
    )
    active = (
        jnp.ones(coarse_shape, dtype=bool) if restrictions is None else restrictions[2]
    )
    np.testing.assert_allclose(
        actual_diagonal[active],
        expected_diagonal[active],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("n_dims", [2, 3])
def test_recursive_materialization_matches_direct_fine_projection(n_dims):
    elements = np.full(n_dims, 4, dtype=int)
    grid = topax.construct_grid(
        n_levels=3,
        sizes=elements.astype(np.float64),
        elements=elements,
    )
    elastic = topax.construct_elastic(1.0, 0.3, 1.0)
    local_stiffness = topax.construct_local_stiffness(grid.fine_level, elastic)
    fine_weighing = jnp.linspace(
        0.1,
        1.0,
        grid.fine_level.n_elements,
        dtype=jnp.float64,
    ).reshape(tuple(elements))

    level_one_operators = galerkin.construct_galerkin_local_operators(
        local_stiffness,
        n_subcells_per_axis=2,
        n_dims=n_dims,
    )
    level_one_matrices = galerkin.materialize_galerkin_element_matrices(
        fine_weighing,
        level_one_operators,
    )
    actual = galerkin.coarsen_heterogeneous_element_matrices(
        grid.levels[1],
        level_one_matrices,
    )

    level_two_operators = galerkin.construct_galerkin_local_operators(
        local_stiffness,
        n_subcells_per_axis=4,
        n_dims=n_dims,
    )
    expected = galerkin.materialize_galerkin_element_matrices(
        fine_weighing,
        level_two_operators,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_incompatible_galerkin_restrictions_are_rejected(n_dims):
    elements = np.full(n_dims, 4, dtype=int)
    grid = topax.construct_grid(
        n_levels=3,
        sizes=np.ones(n_dims),
        elements=elements,
    )
    elastic = topax.construct_elastic(1.0, 0.3, 1.0)
    non_aligned_point = {
        "center": [0.25] + [0.5] * (n_dims - 1),
        "size": [0.0] * n_dims,
    }

    restrictions = list(
        topax.construct_restrictions(
            grid,
            elastic,
            [[non_aligned_point] for _ in range(n_dims)],
        )
    )

    # Deliberately violate Galerkin compatibility.
    restrictions[1] = jnp.ones_like(restrictions[1])

    with pytest.raises(ValueError, match="incompatible with exact Galerkin"):
        topax.solver.setup_galerkin_physics_solver(
            structure=None,
            grid=grid,
            restrictions=tuple(restrictions),
        )
