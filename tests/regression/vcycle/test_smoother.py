import numpy as np
import pytest

import jax
import jax.numpy as jnp

import topax
from topax import vcycle as vcycle_module


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n_smoothing_steps", [1, 2, 4])
def test_jacobi_smooth_n_matches_explicit_iteration(n_smoothing_steps):
    operator = jnp.array(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=jnp.float64,
    )
    right_hand_side = jnp.array([1.0, -2.0], dtype=jnp.float64)
    initial = jnp.array([0.25, -0.5], dtype=jnp.float64)
    inverse_diagonal = 1.0 / jnp.diag(operator)
    damping = 0.6

    def apply_operator(value):
        return operator @ value

    expected = initial
    for _ in range(n_smoothing_steps):
        residual = right_hand_side - apply_operator(expected)
        expected = topax.jacobi_smooth(
            expected,
            residual,
            inverse_diagonal,
            damping,
        )

    smooth = jax.jit(
        lambda solution: topax.jacobi_smooth_n(
            solution,
            right_hand_side,
            inverse_diagonal,
            damping,
            apply_operator,
            n_smoothing_steps,
        )
    )
    actual = smooth(initial)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_jacobi_smooth_n_accepts_known_initial_residual():
    operator = jnp.array(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=jnp.float64,
    )
    right_hand_side = jnp.array([1.0, -2.0], dtype=jnp.float64)
    initial = jnp.zeros_like(right_hand_side)
    inverse_diagonal = 1.0 / jnp.diag(operator)

    def apply_operator(value):
        return operator @ value

    explicit = topax.jacobi_smooth_n(
        initial,
        right_hand_side,
        inverse_diagonal,
        0.6,
        apply_operator,
        4,
    )
    optimized = topax.jacobi_smooth_n(
        initial,
        right_hand_side,
        inverse_diagonal,
        0.6,
        apply_operator,
        4,
        initial_residual=right_hand_side,
    )

    np.testing.assert_allclose(optimized, explicit, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    "n_smoothing_steps",
    [0, -1, 1.5, True, jnp.asarray(2)],
)
def test_jacobi_smooth_n_rejects_invalid_step_counts(n_smoothing_steps):
    with pytest.raises(ValueError, match="positive integer"):
        topax.jacobi_smooth_n(
            jnp.zeros(2),
            jnp.ones(2),
            jnp.ones(2),
            0.6,
            lambda value: value,
            n_smoothing_steps,
        )


def test_vcycle_accepts_custom_smoother(monkeypatch):
    grid = topax.construct_grid(
        n_levels=2,
        sizes=np.array([2.0, 2.0]),
        elements=np.array([2, 2]),
    )
    right_hand_side = jnp.ones((1, 3, 3), dtype=jnp.float64)
    calls = []

    monkeypatch.setattr(
        vcycle_module,
        "solve_sparse",
        lambda solver, coarse_structure, rhs: (solver, rhs),
    )

    def smoother_impl(*args, initial_residual=None, **kwargs):
        calls.append(initial_residual is not None)
        return topax.jacobi_smooth_n(
            *args,
            initial_residual=initial_residual,
            **kwargs,
        )

    def identity_matvec(level, value, local_operator, weighing, restriction):
        return value

    _, solution = vcycle_module.vcycle(
        None,
        right_hand_side,
        grid=grid,
        local_operators=(None, None),
        weighings=(None,),
        inverse_diagonals=(jnp.ones_like(right_hand_side),),
        coarse_structure=None,
        damping=1.0,
        n_smoothing_steps=2,
        smoother_impl=smoother_impl,
        level_matvec_impls=(identity_matvec,),
    )

    assert calls == [True, False]
    np.testing.assert_allclose(solution, right_hand_side)
