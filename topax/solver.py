from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

try:
    import xolky
except ImportError as exc:
    xolky = None
    _XOLKY_IMPORT_ERROR = exc
else:
    _XOLKY_IMPORT_ERROR = None

from .sparse import (
    compute_heterogeneous_physics_csr_values,
    compute_physics_csr_values,
    unpack_sparse_vector,
)


class PhysicsSolverState(NamedTuple):
    physics: Any


class CGFilterSolverState(NamedTuple):
    physics: Any
    helmholtz: Any


def _require_xolky():
    if xolky is None:
        raise ImportError(
            "Xolky is required for sparse solves. Install Xolky manually "
            "before using topax.solver."
        ) from _XOLKY_IMPORT_ERROR
    return xolky


def setup_sparse_solver(structure):
    """Allocate and symbolically analyze a solver outside JAX transformations."""

    xolky_api = _require_xolky()
    device = next(iter(structure.csr_indices.devices()))
    if device.platform == "cpu":
        return xolky_api.setup(
            structure.csr_indices,
            structure.csr_indptr,
            ordering="auto",
            factorization="auto",
        )

    return xolky_api.setup(structure.csr_indices, structure.csr_indptr)


def setup_physics_solver(structure):
    return setup_sparse_solver(structure)


def setup_galerkin_physics_solver(structure, grid, restrictions):
    """Validate Galerkin restrictions and allocate the coarse solver."""

    from .boundary import validate_galerkin_restrictions

    validate_galerkin_restrictions(grid, restrictions)
    return setup_physics_solver(structure)


def setup_helmholtz_solver(structure):
    solver = setup_sparse_solver(structure)
    try:
        factorized = refactor_sparse_solver(solver, structure.csr_data)
        factorized.sequence.block_until_ready()
        return factorized
    except Exception:
        solver.close()
        raise


def refactor_sparse_solver(solver, values):
    """Explicitly convert Topax numeric data at the FP64 Xolky boundary."""

    xolky_api = _require_xolky()
    with jax.enable_x64():
        values = jnp.asarray(values, dtype=jnp.float64)
        return xolky_api.refactor(solver, values)


def refactor_physics_solver(
    solver,
    structure,
    local_material_property,
    weighing,
):
    values = compute_physics_csr_values(
        structure,
        local_material_property,
        weighing,
    )
    return refactor_sparse_solver(solver, values)


def refactor_heterogeneous_physics_solver(solver, structure, element_matrices):
    values = compute_heterogeneous_physics_csr_values(
        structure,
        element_matrices,
    )
    return refactor_sparse_solver(solver, values)


def solve_sparse(solver, structure, right_hand_side):
    """Solve and return both the latest linear solver state and shaped result."""

    xolky_api = _require_xolky()
    result_dtype = jnp.asarray(right_hand_side).dtype
    with jax.enable_x64():
        rhs = jnp.asarray(right_hand_side, dtype=jnp.float64).reshape(-1)
        solver, result = xolky_api.solve(solver, rhs)
    result = unpack_sparse_vector(structure, result)
    return solver, result.astype(result_dtype)


def setup_physics_solver_state(physics_structure, grid, restrictions):
    return PhysicsSolverState(
        physics=setup_galerkin_physics_solver(
            physics_structure,
            grid,
            restrictions,
        )
    )


def setup_cg_filter_solver_state(
    physics_structure,
    helmholtz_structure,
    grid,
    restrictions,
):
    physics = setup_galerkin_physics_solver(
        physics_structure,
        grid,
        restrictions,
    )
    try:
        helmholtz = setup_helmholtz_solver(helmholtz_structure)
    except Exception:
        physics.close()
        raise
    return CGFilterSolverState(physics=physics, helmholtz=helmholtz)


def close_solver_state(solvers):
    """Close the latest solver states in a Topax resource bundle."""

    xolky_api = _require_xolky()
    for solver in jax.tree.leaves(
        solvers,
        is_leaf=lambda value: isinstance(value, xolky_api.SparseCholesky),
    ):
        solver.close()
