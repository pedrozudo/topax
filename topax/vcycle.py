from __future__ import annotations

import functools
import numbers

import jax
import jax.numpy as jnp

from .cg import cg_solve_stateful
from .operator import (
    DEFAULT_SLAB_SIZE,
    matrix_free_diagonal,
    matrix_free_matvec,
    physics_weighing,
)
from .galerkin import (
    DEFAULT_GALERKIN_SLAB_SIZE,
    coarsen_heterogeneous_element_matrices,
    construct_galerkin_local_operators,
    galerkin_diagonal,
    galerkin_matvec,
    heterogeneous_diagonal,
    heterogeneous_matvec,
    materialize_galerkin_element_matrices,
)
from .solver import refactor_heterogeneous_physics_solver, solve_sparse
from .smoother import jacobi_smooth_n


def contract_densities(rho, n_dims):
    window_dimensions = (2,) * n_dims
    rho = jnp.asarray(rho)
    if not jnp.issubdtype(rho.dtype, jnp.inexact):
        rho = rho.astype(jnp.float32)

    if all(size % 2 == 0 for size in rho.shape):
        sums = jax.lax.reduce_window(
            rho,
            jnp.array(0.0, dtype=rho.dtype),
            jax.lax.add,
            window_dimensions,
            window_dimensions,
            "VALID",
        )
        return sums / 2**n_dims

    sums = jax.lax.reduce_window(
        rho,
        jnp.array(0.0, dtype=rho.dtype),
        jax.lax.add,
        window_dimensions,
        window_dimensions,
        "SAME",
    )
    counts = jax.lax.reduce_window(
        jnp.ones_like(rho),
        jnp.array(0.0, dtype=rho.dtype),
        jax.lax.add,
        window_dimensions,
        window_dimensions,
        "SAME",
    )
    return sums / counts


def prolongation_kernel(n_dims, dtype):
    interpolation_weights = jnp.asarray([1.0, 2.0, 1.0], dtype=dtype)
    if n_dims == 2:
        kernel = jnp.einsum("i,j->ij", interpolation_weights, interpolation_weights)
        return (kernel / 4.0)[None, None, ...]
    if n_dims == 3:
        kernel = jnp.einsum(
            "i,j,k->ijk",
            interpolation_weights,
            interpolation_weights,
            interpolation_weights,
        )
        return (kernel / 8.0)[None, None, ...]
    raise NotImplementedError(f"Unsupported V-cycle dimension: {n_dims}")


def _apply_restriction_mask(value, restrictions, level_index):
    if restrictions is None:
        return value
    return value * restrictions[level_index]


def restrict_residual(value, grid, restrictions, level_index):
    value = _apply_restriction_mask(value, restrictions, level_index)
    kernel = prolongation_kernel(grid.n_dims, value.dtype)
    padding = tuple(
        (1, 1 + int(elements) % 2)
        for elements in grid.levels[level_index].elements
    )
    value = jax.lax.conv_with_general_padding(
        value[:, None, ...],
        kernel,
        window_strides=(2,) * grid.n_dims,
        padding=padding,
        lhs_dilation=None,
        rhs_dilation=None,
    )[:, 0, ...]
    return _apply_restriction_mask(value, restrictions, level_index + 1)


def prolongate_correction(value, grid, restrictions, level_index):
    value = _apply_restriction_mask(value, restrictions, level_index + 1)
    kernel = prolongation_kernel(grid.n_dims, value.dtype)
    padding = tuple(
        (1, 1 - int(elements) % 2)
        for elements in grid.levels[level_index].elements
    )
    value = jax.lax.conv_with_general_padding(
        value[:, None, ...],
        kernel,
        window_strides=(1,) * grid.n_dims,
        padding=padding,
        lhs_dilation=(2,) * grid.n_dims,
        rhs_dilation=None,
    )[:, 0, ...]
    return _apply_restriction_mask(value, restrictions, level_index)


def vcycle(
    solver,
    right_hand_side,
    *,
    grid,
    local_operators,
    weighings,
    inverse_diagonals,
    coarse_structure,
    damping,
    n_smoothing_steps=1,
    smoother_impl=jacobi_smooth_n,
    restrictions=None,
    level_matvec_impls=None,
    level_index=0,
):
    """Apply one geometric-multigrid V-cycle.

    ``smoother_impl`` must follow the :func:`jacobi_smooth_n` calling
    convention. When the V-cycle is used as a CG preconditioner, custom
    smoothers should preserve linearity and compatible pre/post smoothing so
    that the preconditioner remains symmetric positive definite.
    """

    if level_index == grid.n_levels - 1:
        return solve_sparse(solver, coarse_structure, right_hand_side)

    level = grid.levels[level_index]
    restriction = None if restrictions is None else restrictions[level_index]
    level_matvec_impl = (
        matrix_free_matvec
        if level_matvec_impls is None
        else level_matvec_impls[level_index]
    )

    def apply_operator(value):
        return level_matvec_impl(
            level,
            value,
            local_operators[level_index],
            weighings[level_index],
            restriction,
        )

    solution = smoother_impl(
        jnp.zeros_like(right_hand_side),
        right_hand_side,
        inverse_diagonals[level_index],
        damping,
        apply_operator,
        n_smoothing_steps,
        initial_residual=right_hand_side,
    )
    residual = right_hand_side - apply_operator(solution)
    coarse_residual = restrict_residual(
        residual,
        grid,
        restrictions,
        level_index,
    )
    solver, coarse_correction = vcycle(
        solver,
        coarse_residual,
        grid=grid,
        local_operators=local_operators,
        weighings=weighings,
        inverse_diagonals=inverse_diagonals,
        coarse_structure=coarse_structure,
        damping=damping,
        n_smoothing_steps=n_smoothing_steps,
        smoother_impl=smoother_impl,
        restrictions=restrictions,
        level_matvec_impls=level_matvec_impls,
        level_index=level_index + 1,
    )
    solution = solution + prolongate_correction(
        coarse_correction,
        grid,
        restrictions,
        level_index,
    )
    solution = smoother_impl(
        solution,
        right_hand_side,
        inverse_diagonals[level_index],
        damping,
        apply_operator,
        n_smoothing_steps,
    )
    solution = _apply_restriction_mask(solution, restrictions, level_index)
    return solver, solution


def _galerkin_matvec_impl(
    level,
    value,
    galerkin_local_operators,
    fine_weighing,
    restrictions=None,
    *,
    slab_size,
):
    return galerkin_matvec(
        level,
        value,
        fine_weighing,
        galerkin_local_operators,
        restrictions,
        slab_size=slab_size,
    )


def _heterogeneous_matvec_impl(
    level,
    value,
    element_matrices,
    unused_weighing,
    restrictions=None,
):
    del unused_weighing
    return heterogeneous_matvec(
        level,
        value,
        element_matrices,
        restrictions,
    )


def _normalize_galerkin_crossover_level(grid, galerkin_crossover_level):
    if galerkin_crossover_level is None:
        return grid.n_levels - 1
    if isinstance(galerkin_crossover_level, (bool, jax.Array)) or not isinstance(
        galerkin_crossover_level,
        numbers.Integral,
    ):
        raise ValueError(
            "galerkin_crossover_level must be a Python integer or None."
        )
    galerkin_crossover_level = int(galerkin_crossover_level)
    if not 1 <= galerkin_crossover_level < grid.n_levels:
        raise ValueError(
            "galerkin_crossover_level must satisfy "
            f"1 <= level < {grid.n_levels}."
        )
    return galerkin_crossover_level


def _normalize_galerkin_slab_size(galerkin_slab_size):
    if (
        isinstance(galerkin_slab_size, bool)
        or not isinstance(galerkin_slab_size, numbers.Integral)
        or galerkin_slab_size < 1
    ):
        raise ValueError("galerkin_slab_size must be a positive integer.")
    return int(galerkin_slab_size)


def construct_physics_multigrid_data(
    solver,
    grid,
    fine_local_operator,
    coarse_structure,
    rho,
    simp_penalty,
    rho_min,
    galerkin_crossover_level=None,
    galerkin_slab_size=DEFAULT_GALERKIN_SLAB_SIZE,
):
    """Construct an exact dyadic Galerkin elasticity hierarchy.

    Levels below ``galerkin_crossover_level`` use the implicit fixed-``G_s``
    representation. The crossover and all deeper levels use heterogeneous
    materialized element matrices. ``None`` selects the coarsest level and
    therefore preserves the original reference implementation.
    """

    if grid.n_levels < 2:
        raise ValueError("A V-cycle requires at least two grid levels.")
    galerkin_crossover_level = _normalize_galerkin_crossover_level(
        grid,
        galerkin_crossover_level,
    )
    galerkin_slab_size = _normalize_galerkin_slab_size(galerkin_slab_size)

    coarsest_factor = 2 ** (grid.n_levels - 1)
    if any(int(elements) % coarsest_factor != 0 for elements in grid.elements):
        raise ValueError(
            "Galerkin elasticity currently requires a fully dyadic hierarchy; "
            "fine-grid element counts must be divisible by "
            f"{coarsest_factor}."
        )

    fine_weighing = physics_weighing(
        jax.lax.stop_gradient(rho),
        simp_penalty,
        rho_min,
    )
    hierarchy_operators = [fine_local_operator]
    weighings = [fine_weighing]
    inverse_diagonals = [
        1.0
        / matrix_free_diagonal(
            grid.fine_level,
            fine_local_operator,
            fine_weighing,
        )
    ]

    for level_index in range(1, galerkin_crossover_level):
        level = grid.levels[level_index]
        galerkin_local_operators = construct_galerkin_local_operators(
            fine_local_operator,
            n_subcells_per_axis=2**level_index,
            n_dims=grid.n_dims,
        )
        diagonal = galerkin_diagonal(
            level,
            fine_weighing,
            galerkin_local_operators,
            slab_size=galerkin_slab_size,
        )
        hierarchy_operators.append(galerkin_local_operators)
        weighings.append(fine_weighing)
        inverse_diagonals.append(1.0 / diagonal)

    crossover_galerkin_local_operators = construct_galerkin_local_operators(
        fine_local_operator,
        n_subcells_per_axis=2**galerkin_crossover_level,
        n_dims=grid.n_dims,
    )
    materialized_element_matrices = materialize_galerkin_element_matrices(
        fine_weighing,
        crossover_galerkin_local_operators,
    )

    for level_index in range(
        galerkin_crossover_level,
        grid.n_levels - 1,
    ):
        level = grid.levels[level_index]
        diagonal = heterogeneous_diagonal(
            level,
            materialized_element_matrices,
        )
        hierarchy_operators.append(materialized_element_matrices)
        weighings.append(None)
        inverse_diagonals.append(1.0 / diagonal)
        materialized_element_matrices = coarsen_heterogeneous_element_matrices(
            level,
            materialized_element_matrices,
        )

    solver = refactor_heterogeneous_physics_solver(
        solver,
        coarse_structure,
        materialized_element_matrices,
    )
    return (
        solver,
        tuple(hierarchy_operators),
        tuple(weighings),
        tuple(inverse_diagonals),
    )


def construct_helmholtz_multigrid_data(
    grid,
    local_operators,
    rho_mask,
):
    if grid.n_levels < 2:
        raise ValueError("A V-cycle requires at least two grid levels.")
    if len(local_operators) != grid.n_levels:
        raise ValueError("Expected one local operator per grid level.")

    mask_level = rho_mask
    weighings = []
    inverse_diagonals = []
    for level, local_operator in zip(grid.levels[:-1], local_operators[:-1]):
        diagonal = matrix_free_diagonal(level, local_operator, mask_level)
        weighings.append(mask_level)
        inverse_diagonals.append(1.0 / diagonal)
        mask_level = contract_densities(mask_level, grid.n_dims)
    return tuple(weighings), tuple(inverse_diagonals)


def physics_vcycle_solve(
    solver,
    grid,
    fine_local_operator,
    coarse_structure,
    rho,
    forces,
    restrictions,
    x0,
    simp_penalty,
    rho_min,
    damping,
    tol,
    maxiter,
    n_smoothing_steps=1,
    smoother_impl=jacobi_smooth_n,
    galerkin_crossover_level=None,
    galerkin_slab_size=DEFAULT_GALERKIN_SLAB_SIZE,
    slab_size=DEFAULT_SLAB_SIZE,
    slab_axis=None,
):
    """Solve elasticity with an index-free implicit Galerkin hierarchy."""
    galerkin_crossover_level = _normalize_galerkin_crossover_level(
        grid,
        galerkin_crossover_level,
    )
    galerkin_slab_size = _normalize_galerkin_slab_size(galerkin_slab_size)
    (
        solver,
        hierarchy_operators,
        weighings,
        inverse_diagonals,
    ) = construct_physics_multigrid_data(
        solver,
        grid,
        fine_local_operator,
        coarse_structure,
        rho,
        simp_penalty,
        rho_min,
        galerkin_crossover_level=galerkin_crossover_level,
        galerkin_slab_size=galerkin_slab_size,
    )

    fine_matvec_impl = functools.partial(
        matrix_free_matvec,
        slab_size=slab_size,
        slab_axis=slab_axis,
    )

    def matvec(value):
        return fine_matvec_impl(
            grid.fine_level,
            value,
            hierarchy_operators[0],
            weighings[0],
            restrictions[0],
        )

    implicit_galerkin_matvec_impl = functools.partial(
        _galerkin_matvec_impl,
        slab_size=galerkin_slab_size,
    )
    implicit_level_matvec_impls = [
        implicit_galerkin_matvec_impl
        for _ in range(1, galerkin_crossover_level)
    ]
    level_matvec_impls = (
        (fine_matvec_impl,)
        + tuple(implicit_level_matvec_impls)
        + (_heterogeneous_matvec_impl,)
        * (grid.n_levels - 1 - galerkin_crossover_level)
    )
    precondition = functools.partial(
        vcycle,
        grid=grid,
        local_operators=hierarchy_operators,
        weighings=weighings,
        inverse_diagonals=inverse_diagonals,
        coarse_structure=coarse_structure,
        damping=damping,
        n_smoothing_steps=n_smoothing_steps,
        smoother_impl=smoother_impl,
        restrictions=restrictions,
        level_matvec_impls=level_matvec_impls,
    )
    return cg_solve_stateful(
        solver,
        matvec,
        forces,
        x0,
        precondition=precondition,
        tol=tol,
        maxiter=maxiter,
    )


def helmholtz_vcycle_solve(
    solver,
    grid,
    local_operators,
    coarse_structure,
    rho_mask,
    rho_nodal,
    x0,
    damping,
    tol,
    maxiter,
    n_smoothing_steps=1,
    smoother_impl=jacobi_smooth_n,
    slab_size=DEFAULT_SLAB_SIZE,
    slab_axis=None,
):
    weighings, inverse_diagonals = construct_helmholtz_multigrid_data(
        grid,
        local_operators,
        rho_mask,
    )

    fine_matvec_impl = functools.partial(
        matrix_free_matvec,
        slab_size=slab_size,
        slab_axis=slab_axis,
    )

    def matvec(value):
        return fine_matvec_impl(
            grid.fine_level,
            value,
            local_operators[0],
            weighings[0],
        )

    precondition = functools.partial(
        vcycle,
        grid=grid,
        local_operators=local_operators,
        weighings=weighings,
        inverse_diagonals=inverse_diagonals,
        coarse_structure=coarse_structure,
        damping=damping,
        n_smoothing_steps=n_smoothing_steps,
        smoother_impl=smoother_impl,
        restrictions=None,
        level_matvec_impls=(fine_matvec_impl,)
        + (matrix_free_matvec,) * (grid.n_levels - 2),
    )
    return cg_solve_stateful(
        solver,
        matvec,
        rho_nodal,
        x0,
        precondition=precondition,
        tol=tol,
        maxiter=maxiter,
    )


def construct_physics_solve(**configuration):
    return functools.partial(physics_vcycle_solve, **configuration)


def construct_helmholtz_solve(**configuration):
    return functools.partial(helmholtz_vcycle_solve, **configuration)
