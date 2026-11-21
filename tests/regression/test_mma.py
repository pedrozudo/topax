import jax
import jax.numpy as jnp
import numpy as np

import topax


def _mma_step(x, constraint_value, *, move, lam_max=1e8):
    params = topax.optimizer.mma.init_mma_1con_params(
        move=move,
        lam_max=lam_max,
    )
    state = topax.optimizer.mma.init_mma_1con_state(x, params)
    step = jax.jit(topax.optimizer.mma.construct_mma_step(params))
    return step(
        x,
        jnp.asarray(1.0, dtype=x.dtype),
        -jnp.ones_like(x),
        jnp.asarray(constraint_value, dtype=x.dtype),
        jnp.full_like(x, 1.0 / x.size),
        state,
    )


def test_mma_reports_feasible_bracketed_subproblem():
    x = jnp.full(4, 0.5, dtype=jnp.float32)

    x_new, _, info = _mma_step(x, 0.1, move=0.2)

    assert bool(info["bracketed"])
    assert bool(info["feasible"])
    assert float(info["phi"]) <= 0.0
    assert np.all(np.asarray(x_new) < np.asarray(x))


def test_mma_reports_move_limited_infeasible_subproblem():
    x = jnp.full(4, 0.9, dtype=jnp.float32)
    lam_max = 1e4

    x_new, _, info = _mma_step(x, 0.5, move=0.01, lam_max=lam_max)

    assert not bool(info["bracketed"])
    assert not bool(info["feasible"])
    assert float(info["phi"]) > 0.0
    assert float(info["lam"]) == lam_max
    np.testing.assert_allclose(x_new, 0.89, rtol=0.0, atol=1e-6)
