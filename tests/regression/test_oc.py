import jax
import jax.numpy as jnp
import numpy as np

import topax


def test_oc_default_update_matches_wang_benchmark():
    x = jnp.array([0.08, 0.10, 0.13, 0.17])
    objective_grad = jnp.array([-1.0, -2.0, -0.5, -3.0])
    params = topax.optimizer.oc.init_oc_params(
        move=0.2,
        bisect_tol=1.0e-6,
    )
    step = jax.jit(topax.optimizer.oc.construct_oc_step(params))

    x_new, info = step(
        x,
        objective_grad,
        jnp.mean(x) - 0.12,
        jnp.ones_like(x),
    )

    np.testing.assert_allclose(
        x_new,
        [0.06317932, 0.11168631, 0.07259610, 0.23253830],
        rtol=1.0e-6,
    )
    assert float(info["change"]) == np.max(np.abs(np.asarray(x_new - x)))
    assert float(info["constraint"]) <= 1.0e-6


def test_oc_uses_constraint_gradient_in_update():
    x = jnp.array([0.06, 0.08, 0.11, 0.15])
    objective_grad = jnp.array([-1.2, -0.8, -2.0, -0.4])
    constraint_grad = jnp.full_like(x, 0.25)
    params = topax.optimizer.oc.init_oc_params(
        move=0.2,
        bisect_tol=1.0e-6,
    )

    x_new, _ = topax.optimizer.oc.oc_step(
        x,
        objective_grad,
        jnp.mean(x) - 0.1,
        constraint_grad,
        params=params,
    )

    np.testing.assert_allclose(
        x_new,
        [0.06780970, 0.07382185, 0.16049357, 0.09787487],
        rtol=1.0e-6,
    )


def test_oc_fixed_bisection_matches_lazarov_benchmark():
    x = jnp.array([0.35, 0.45, 0.55, 0.65])
    objective_grad = jnp.array([-0.8, -1.6, -0.4, -2.0])
    params = topax.optimizer.oc.init_oc_params(
        xmin=1.0e-3,
        move=0.2,
        max_bisect=64,
        bisect_tol=0.0,
    )

    x_new, info = topax.optimizer.oc.oc_step(
        x,
        objective_grad,
        jnp.sum(x) - 0.5 * x.size,
        jnp.ones_like(x),
        params=params,
    )

    np.testing.assert_allclose(
        x_new,
        [0.28672339, 0.52134185, 0.35, 0.84193476],
        rtol=1.0e-6,
    )
    assert float(info["change"]) == np.max(np.abs(np.asarray(x_new - x)))
    np.testing.assert_allclose(info["lam"], 1.19206471, rtol=1.0e-6)
