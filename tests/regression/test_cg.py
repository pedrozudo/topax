import jax.numpy as jnp
import numpy as np

from topax.cg import cg_solve


def test_cg_reports_true_residual_and_restarts_after_residual_drift():
    n = 32
    diagonal = jnp.logspace(0.0, 4.0, n, dtype=jnp.float32)
    right_hand_side = jnp.cos(jnp.arange(n, dtype=jnp.float32))

    solution, info = cg_solve(
        lambda value: diagonal * value,
        right_hand_side,
        tol=1e-10,
        maxiter=1000,
    )

    true_residual = right_hand_side - diagonal * solution
    true_error = jnp.vdot(true_residual, true_residual).real
    relative_residual = jnp.linalg.norm(true_residual) / jnp.linalg.norm(
        right_hand_side
    )

    np.testing.assert_allclose(info["error"], true_error, rtol=3e-2, atol=1e-15)
    np.testing.assert_allclose(
        info["relative_residual"],
        relative_residual,
        rtol=2e-2,
    )
    assert int(info["n_restarts"]) > 0
    assert float(info["recursive_error"]) < float(info["error"])
    assert not bool(info["converged"])
    assert not bool(info["breakdown"])


def test_cg_reports_zero_denominator_breakdown():
    right_hand_side = jnp.ones(4, dtype=jnp.float32)

    solution, info = cg_solve(
        lambda value: jnp.zeros_like(value),
        right_hand_side,
        tol=1e-5,
        maxiter=10,
    )

    np.testing.assert_array_equal(solution, jnp.zeros_like(right_hand_side))
    assert bool(info["breakdown"])
    assert not bool(info["converged"])
    assert int(info["n_iter"]) == 0
    assert float(info["relative_residual"]) == 1.0
