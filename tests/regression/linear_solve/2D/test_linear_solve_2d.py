import importlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import topax

_problems = importlib.import_module(
    "tests.regression.linear_solve.2D._problems"
)
_references = importlib.import_module(
    "tests.regression.linear_solve.2D._references"
)

RHO_MIN = _problems.RHO_MIN
SIMP_PENALTY = _problems.SIMP_PENALTY
construct_linear_problem = _problems.construct_linear_problem
LINEAR_SOLVE_REFERENCES = _references.LINEAR_SOLVE_REFERENCES


jax.config.update(
    "jax_enable_x64",
    True,
)


# =============================================================================
# Solver configuration
# =============================================================================

SOLVE_TOL = 1e-10
MAXITER = 10_000

DAMPING = 0.8


# =============================================================================
# Comparison tolerances
#
# The reference values come from the independent Julia implementation.
#
# We deliberately use a slightly looser tolerance than machine precision:
# iterative solves need not produce bitwise-identical extrema, and the
# uniform-density matrix-free path can differ slightly in floating-point
# evaluation.
# =============================================================================

REFERENCE_RTOL = 1e-7
REFERENCE_ATOL = 1e-10


# =============================================================================
# Parameter sets
# =============================================================================


GRID_CASES = [
    pytest.param(
        nx,
        ny,
        id=f"{nx}x{ny}",
    )
    for nx, ny in LINEAR_SOLVE_REFERENCES
]


def construct_vcycle_cases():
    """
    Construct all valid multigrid hierarchies for every reference grid.

    A hierarchy with n_levels requires the element counts to be divisible by

        2 ** (n_levels - 1)

    in every spatial dimension.

    Examples:

        16 x 16:
            L2, L3, L4, L5

        32 x 16:
            L2, L3, L4, L5

        64 x 32:
            L2, L3, L4, L5, L6
    """

    cases = []

    for nx, ny in LINEAR_SOLVE_REFERENCES:
        n_levels = 2

        while True:
            divisor = 2 ** (n_levels - 1)

            if nx % divisor != 0 or ny % divisor != 0:
                break

            cases.append(
                pytest.param(
                    nx,
                    ny,
                    n_levels,
                    id=f"{nx}x{ny}-L{n_levels}",
                )
            )

            n_levels += 1

    return cases


VCYCLE_CASES = construct_vcycle_cases()


# =============================================================================
# Helpers
# =============================================================================


def assert_matches_reference(
    u,
    nx,
    ny,
):
    u = jax.block_until_ready(u)

    reference = LINEAR_SOLVE_REFERENCES[
        (
            nx,
            ny,
        )
    ]

    expected_min = reference["u_min"]
    expected_max = reference["u_max"]

    u_min = float(jnp.min(u))

    u_max = float(jnp.max(u))

    np.testing.assert_allclose(
        u_min,
        expected_min,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
        err_msg=(f"solution minimum differs from legacy reference for {nx}x{ny}"),
    )

    np.testing.assert_allclose(
        u_max,
        expected_max,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
        err_msg=(f"solution maximum differs from legacy reference for {nx}x{ny}"),
    )


# =============================================================================
# Sparse direct solve
# =============================================================================


@pytest.mark.parametrize(
    "nx, ny",
    GRID_CASES,
)
def test_sparse_direct_solve(
    nx,
    ny,
):
    (
        elastic,
        grid,
        local_stiffnesses,
        restrictions,
        forces,
        rho,
    ) = construct_linear_problem(
        elements=(
            nx,
            ny,
        ),
        n_levels=1,
    )

    level = grid.fine_level

    restriction = restrictions[0]

    sparse_structure = topax.sparse.construct_sparse_physics_structure(
        level,
        n_dofs_node=level.n_dims,
        restriction=restriction,
    )

    weighing = topax.operator.simp(
        rho.reshape(-1),
        SIMP_PENALTY,
        RHO_MIN,
    )
    solver = topax.solver.setup_physics_solver(sparse_structure)
    try:
        solver = topax.solver.refactor_physics_solver(
            solver,
            sparse_structure,
            local_stiffnesses[0],
            weighing,
        )

        @jax.jit
        def solve(solver, structure, right_hand_side):
            return topax.solver.solve_sparse(
                solver,
                structure,
                right_hand_side,
            )

        solver, u = solve(
            solver,
            sparse_structure,
            forces,
        )
        assert_matches_reference(u, nx, ny)
    finally:
        solver.close()


# =============================================================================
# Matrix-free CG
# =============================================================================


@pytest.mark.parametrize(
    "nx, ny",
    GRID_CASES,
)
def test_matrix_free_cg_solve(
    nx,
    ny,
):
    (
        elastic,
        grid,
        local_stiffnesses,
        restrictions,
        forces,
        rho,
    ) = construct_linear_problem(
        elements=(
            nx,
            ny,
        ),
        n_levels=1,
    )

    level = grid.fine_level

    restriction = restrictions[0]

    weighing = topax.operator.physics_weighing(
        rho,
        SIMP_PENALTY,
        RHO_MIN,
    )

    def Kmatvec(x):
        return topax.operator.matrix_free_matvec(
            level,
            x,
            local_stiffnesses[0],
            weighing,
            restriction,
        )

    diagonal = topax.operator.matrix_free_diagonal(
        level,
        local_stiffnesses[0],
        weighing,
    )

    u, info = topax.cg.cg_solve(
        Kmatvec,
        forces,
        tol=SOLVE_TOL,
        maxiter=MAXITER,
        M=lambda x: x / diagonal,
    )

    u = jax.block_until_ready(u)

    assert bool(info["converged"])

    assert_matches_reference(
        u,
        nx,
        ny,
    )


# =============================================================================
# CG + V-cycle
#
# The expected solution depends only on the physical FE discretization.
# n_levels changes the preconditioner, not the linear system being solved.
#
# Therefore every valid hierarchy for a given (nx, ny) is compared with the
# exact same Julia reference values.
# =============================================================================


@pytest.mark.parametrize(
    "nx, ny, n_levels",
    VCYCLE_CASES,
)
def test_vcycle_cg_solve(
    nx,
    ny,
    n_levels,
):
    (
        elastic,
        grid,
        local_stiffnesses,
        restrictions,
        forces,
        rho,
    ) = construct_linear_problem(
        elements=(
            nx,
            ny,
        ),
        n_levels=n_levels,
    )

    coarse_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )

    x0 = jnp.zeros_like(forces)
    solver = topax.solver.setup_galerkin_physics_solver(
        coarse_structure, grid, restrictions
    )

    @jax.jit
    def solve(solver, rho, forces, restrictions, x0):
        return topax.vcycle.physics_vcycle_solve(
            solver=solver,
            grid=grid,
            fine_local_operator=local_stiffnesses[0],
            coarse_structure=coarse_structure,
            rho=rho,
            forces=forces,
            restrictions=restrictions,
            x0=x0,
            simp_penalty=SIMP_PENALTY,
            rho_min=RHO_MIN,
            damping=DAMPING,
            tol=SOLVE_TOL,
            maxiter=MAXITER,
        )

    try:
        solver, u, info = solve(
            solver,
            rho,
            forces,
            restrictions,
            x0,
        )
        u = jax.block_until_ready(u)
        assert bool(info["converged"])
        assert_matches_reference(u, nx, ny)
    finally:
        solver.close()


@pytest.mark.parametrize("n_smoothing_steps", [1, 2, 4])
def test_vcycle_cg_solve_multiple_smoothing_steps(n_smoothing_steps):
    nx, ny = min(
        LINEAR_SOLVE_REFERENCES,
        key=lambda elements: elements[0] * elements[1],
    )
    (
        elastic,
        grid,
        local_stiffnesses,
        restrictions,
        forces,
        rho,
    ) = construct_linear_problem(
        elements=(nx, ny),
        n_levels=2,
    )

    coarse_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )
    x0 = jnp.zeros_like(forces)
    solver = topax.solver.setup_galerkin_physics_solver(
        coarse_structure, grid, restrictions
    )

    @jax.jit
    def solve(solver, rho, forces, restrictions, x0):
        return topax.vcycle.physics_vcycle_solve(
            solver=solver,
            grid=grid,
            fine_local_operator=local_stiffnesses[0],
            coarse_structure=coarse_structure,
            rho=rho,
            forces=forces,
            restrictions=restrictions,
            x0=x0,
            simp_penalty=SIMP_PENALTY,
            rho_min=RHO_MIN,
            damping=DAMPING,
            tol=SOLVE_TOL,
            maxiter=MAXITER,
            n_smoothing_steps=n_smoothing_steps,
        )

    try:
        solver, u, info = solve(
            solver,
            rho,
            forces,
            restrictions,
            x0,
        )
        u = jax.block_until_ready(u)
        assert bool(info["converged"])
        assert_matches_reference(u, nx, ny)
    finally:
        solver.close()
