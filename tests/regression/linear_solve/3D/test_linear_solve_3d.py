import importlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import topax

_problems = importlib.import_module("tests.regression.linear_solve.3D._problems")
_references = importlib.import_module("tests.regression.linear_solve.3D._references")

RHO_MIN = _problems.RHO_MIN
SIMP_PENALTY = _problems.SIMP_PENALTY
construct_linear_problem_3d = _problems.construct_linear_problem_3d
LINEAR_SOLVE_REFERENCES_3D = _references.LINEAR_SOLVE_REFERENCES_3D


jax.config.update("jax_enable_x64", True)


# Reliable convergence is checked against b - A @ x; the long beam reaches
# a true-residual floor of a few times 1e-10 on CPU.
SOLVE_TOL = 1e-9
MAXITER = 10_000
DAMPING = 0.8

REFERENCE_RTOL = 1e-7
REFERENCE_ATOL = 1e-10


GRID_CASES = [
    pytest.param(*elements, id="x".join(map(str, elements)))
    for elements in LINEAR_SOLVE_REFERENCES_3D
]

def construct_vcycle_cases():
    cases = []
    for elements in LINEAR_SOLVE_REFERENCES_3D:
        max_levels = int(np.floor(np.log2(min(elements)))) + 1
        for n_levels in range(2, max_levels + 1):
            cases.append(
                pytest.param(
                    *elements,
                    n_levels,
                    id=f"{'x'.join(map(str, elements))}-L{n_levels}",
                )
            )
    return cases


VCYCLE_CASES = construct_vcycle_cases()


def assert_matches_reference(u, elements):
    u = jax.block_until_ready(u)
    reference = LINEAR_SOLVE_REFERENCES_3D[elements]

    actual = {
        "u_min": float(jnp.min(u)),
        "u_max": float(jnp.max(u)),
        "u_norm": float(jnp.linalg.norm(u)),
    }

    for quantity, expected in reference.items():
        np.testing.assert_allclose(
            actual[quantity],
            expected,
            rtol=REFERENCE_RTOL,
            atol=REFERENCE_ATOL,
            err_msg=(
                f"solution {quantity} differs from the Julia reference for "
                f"{'x'.join(map(str, elements))}"
            ),
        )


@pytest.mark.parametrize("nx, ny, nz", GRID_CASES)
def test_sparse_direct_solve_3d(nx, ny, nz):
    elements = (nx, ny, nz)
    grid, local_stiffness, restriction, forces, rho = construct_linear_problem_3d(
        elements
    )

    sparse_structure = topax.sparse.construct_sparse_physics_structure(
        grid.fine_level,
        n_dofs_node=grid.n_dims,
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
            local_stiffness,
            weighing,
        )

        @jax.jit
        def solve(solver, structure, right_hand_side):
            return topax.solver.solve_sparse(
                solver,
                structure,
                right_hand_side,
            )

        solver, displacement = solve(
            solver,
            sparse_structure,
            forces,
        )
        assert_matches_reference(displacement, elements)
    finally:
        solver.close()


@pytest.mark.parametrize("nx, ny, nz", GRID_CASES)
def test_matrix_free_cg_solve_3d(nx, ny, nz):
    elements = (nx, ny, nz)
    grid, local_stiffness, restriction, forces, rho = construct_linear_problem_3d(
        elements
    )

    weighing = topax.operator.physics_weighing(
        rho,
        SIMP_PENALTY,
        RHO_MIN,
    )
    diagonal = topax.operator.matrix_free_diagonal(
        grid.fine_level,
        local_stiffness,
        weighing,
    )

    displacement, info = topax.cg.cg_solve(
        lambda x: topax.operator.matrix_free_matvec(
            grid.fine_level,
            x,
            local_stiffness,
            weighing,
            restriction,
        ),
        forces,
        tol=SOLVE_TOL,
        maxiter=MAXITER,
        M=lambda x: x / diagonal,
    )

    displacement = jax.block_until_ready(displacement)

    assert bool(info["converged"])
    assert_matches_reference(displacement, elements)


def construct_multilevel_linear_problem_3d(elements, n_levels):
    elements = np.asarray(elements, dtype=int)
    elastic = topax.construct_elastic(
        _problems.ELASTIC_MODULE,
        _problems.POISSON_RATIO,
    )
    grid = topax.construct_grid(
        n_levels=n_levels,
        sizes=_problems.SIZES,
        elements=elements,
    )
    local_stiffnesses = tuple(
        topax.construct_local_stiffness(level, elastic) for level in grid.levels
    )
    restrictions = topax.construct_restrictions(
        grid, elastic, _problems.DIRICHLET_BOUNDARY_CONDITIONS
    )
    forces = topax.construct_forces(grid, elastic, _problems.CONCENTRATED_FORCES)
    rho = _problems.RHO * jnp.ones(tuple(elements), dtype=jnp.float64)
    return grid, local_stiffnesses, restrictions, forces, rho


@pytest.mark.parametrize("nx, ny, nz, n_levels", VCYCLE_CASES)
def test_vcycle_cg_solve_3d(nx, ny, nz, n_levels):
    elements = (nx, ny, nz)
    grid, local_stiffnesses, restrictions, forces, rho = (
        construct_multilevel_linear_problem_3d(elements, n_levels)
    )
    coarse_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )
    solver = topax.solver.setup_galerkin_physics_solver(
        coarse_structure, grid, restrictions
    )

    @jax.jit
    def solve(solver, rho, forces, restrictions):
        return topax.vcycle.physics_vcycle_solve(
            solver=solver,
            grid=grid,
            fine_local_operator=local_stiffnesses[0],
            coarse_structure=coarse_structure,
            rho=rho,
            forces=forces,
            restrictions=restrictions,
            x0=jnp.zeros_like(forces),
            simp_penalty=SIMP_PENALTY,
            rho_min=RHO_MIN,
            damping=DAMPING,
            tol=SOLVE_TOL,
            maxiter=MAXITER,
        )

    try:
        solver, displacement, info = solve(
            solver,
            rho,
            forces,
            restrictions,
        )
        assert bool(info["converged"])
        assert_matches_reference(displacement, elements)
    finally:
        solver.close()
