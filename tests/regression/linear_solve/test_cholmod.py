import jax
import numpy as np
import pytest


pytest.importorskip(
    "xolky._xolky_cholmod",
    reason="xolky was built without CHOLMOD support",
)

import topax.solver as solver_api
from topax.sparse import HelmholtzCSRStructure, sparse_todense


jax.config.update("jax_enable_x64", True)


def test_cholmod_refactor_and_solve_on_cpu():
    cpu = jax.devices("cpu")[0]

    def cpu_array(values, dtype):
        return jax.device_put(np.asarray(values, dtype=dtype), cpu)

    structure = HelmholtzCSRStructure(
        csr_indices=cpu_array([0, 0, 1], np.int32),
        csr_indptr=cpu_array([0, 1, 3], np.int32),
        csr_data=cpu_array([4.0, 1.0, 3.0], np.float64),
        n_dofs_node=1,
        n_dofs=2,
        vector_shape=(1, 2),
    )
    rhs = cpu_array([1.0, 2.0], np.float64)

    @jax.jit
    def refactor_and_solve(solver, values, right_hand_side):
        solver = solver_api.refactor_sparse_solver(solver, values)
        return solver_api.solve_sparse(solver, structure, right_hand_side)

    solver = solver_api.setup_sparse_solver(structure)
    try:
        assert solver.backend == "cholmod"

        for values in (
            structure.csr_data,
            cpu_array([5.0, 0.5, 2.0], np.float64),
        ):
            solver, solution = refactor_and_solve(solver, values, rhs)
            matrix = sparse_todense(structure, values)
            np.testing.assert_allclose(
                matrix @ solution.reshape(-1),
                rhs,
                rtol=1e-12,
                atol=1e-12,
            )
    finally:
        solver.close()
