import jax.numpy as jnp
from jax import lax


def _squared_norm(value):
    return jnp.vdot(value, value).real


def _all_finite(*values):
    result = jnp.asarray(True)
    for value in values:
        result = jnp.logical_and(result, jnp.all(jnp.isfinite(value)))
    return result


def _nonzero_finite(value):
    return jnp.logical_and(_all_finite(value), jnp.abs(value) > 0.0)


def _positive_finite(value):
    return jnp.logical_and(_all_finite(value), jnp.real(value) > 0.0)


def _relative_residual(error, right_hand_side_norm_squared):
    return jnp.where(
        right_hand_side_norm_squared > 0.0,
        jnp.sqrt(error / right_hand_side_norm_squared),
        jnp.where(error == 0.0, 0.0, jnp.inf),
    )


def _cg_iterations_stateful(
    solver,
    matvec,
    b,
    x0,
    precondition,
    tol,
    atol,
    maxiter,
):
    """Run reliable preconditioned CG while threading solver state.

    A CG segment uses the inexpensive recursively updated residual. Whenever
    that residual reaches the requested tolerance, the true residual
    ``b - matvec(x)`` is evaluated. If the true residual does not pass, a new
    segment restarts from it. Iterations across all segments share ``maxiter``.
    """

    b_norm_squared = _squared_norm(b)
    tolerance_squared = jnp.maximum(
        jnp.square(tol) * b_norm_squared,
        jnp.square(atol),
    )
    residual0 = b - matvec(x0)
    error0 = _squared_norm(residual0)
    finite0 = _all_finite(x0, residual0, error0)
    converged0 = finite0 & (error0 <= tolerance_squared)

    # Outer iterations are residual-replacement restarts. A segment stops when
    # its recursive residual passes, maxiter is reached, or CG breaks down.
    initial = (
        solver,
        x0,
        jnp.asarray(0, dtype=jnp.int32),
        converged0,
        jnp.logical_not(finite0),
        jnp.asarray(0, dtype=jnp.int32),
        error0,
        residual0,
        error0,
    )

    def restart_cond(carry):
        _, _, iteration, converged, breakdown, _, _, _, _ = carry
        return (
            jnp.logical_not(converged)
            & jnp.logical_not(breakdown)
            & (iteration < maxiter)
        )

    def run_segment(carry):
        solver, x, iteration, _, _, restarts, _, residual, true_error = carry
        solver, z = precondition(solver, residual)
        gamma = jnp.vdot(residual, z)
        start_is_valid = _all_finite(residual, z, gamma) & _nonzero_finite(gamma)

        def invalid_start(_):
            return (
                solver,
                x,
                iteration,
                jnp.asarray(False),
                jnp.asarray(True),
                restarts,
                _squared_norm(residual),
                residual,
                true_error,
            )

        def valid_start(_):
            segment_initial = (
                solver,
                x,
                residual,
                gamma,
                z,
                iteration,
                jnp.asarray(False),
            )

            def iteration_cond(segment):
                _, _, residual, _, _, iteration, breakdown = segment
                return (
                    (_squared_norm(residual) > tolerance_squared)
                    & (iteration < maxiter)
                    & jnp.logical_not(breakdown)
                )

            def iteration_body(segment):
                (
                    solver,
                    x,
                    residual,
                    gamma,
                    direction,
                    iteration,
                    _,
                ) = segment
                image = matvec(direction)
                denominator = jnp.vdot(direction, image)

                def denominator_breakdown(_):
                    return (
                        solver,
                        x,
                        residual,
                        gamma,
                        direction,
                        iteration,
                        jnp.asarray(True),
                    )

                def take_step(_):
                    alpha = gamma / denominator
                    x_next = x + alpha * direction
                    residual_next = residual - alpha * image
                    solver_next, z_next = precondition(solver, residual_next)
                    gamma_next = jnp.vdot(residual_next, z_next)
                    beta = gamma_next / gamma
                    direction_next = z_next + beta * direction
                    iteration_next = iteration + 1
                    recursive_error = _squared_norm(residual_next)
                    needs_another_iteration = (
                        (recursive_error > tolerance_squared)
                        & (iteration_next < maxiter)
                    )
                    values_are_finite = _all_finite(
                        alpha,
                        x_next,
                        residual_next,
                        z_next,
                        gamma_next,
                        beta,
                        direction_next,
                        recursive_error,
                    )
                    gamma_is_valid = jnp.logical_or(
                        jnp.logical_not(needs_another_iteration),
                        _nonzero_finite(gamma_next),
                    )
                    breakdown = jnp.logical_not(
                        values_are_finite & gamma_is_valid
                    )
                    return (
                        solver_next,
                        x_next,
                        residual_next,
                        gamma_next,
                        direction_next,
                        iteration_next,
                        breakdown,
                    )

                return lax.cond(
                    _positive_finite(denominator),
                    take_step,
                    denominator_breakdown,
                    operand=None,
                )

            (
                solver_next,
                x_next,
                recursive_residual,
                _,
                _,
                iteration_next,
                breakdown,
            ) = lax.while_loop(
                iteration_cond,
                iteration_body,
                segment_initial,
            )

            true_residual = b - matvec(x_next)
            true_error = _squared_norm(true_residual)
            final_is_finite = _all_finite(x_next, true_residual, true_error)
            breakdown = breakdown | jnp.logical_not(final_is_finite)
            converged = (
                jnp.logical_not(breakdown)
                & final_is_finite
                & (true_error <= tolerance_squared)
            )
            restart = (
                jnp.logical_not(converged)
                & jnp.logical_not(breakdown)
                & (iteration_next < maxiter)
            )
            return (
                solver_next,
                x_next,
                iteration_next,
                converged,
                breakdown,
                restarts + restart.astype(jnp.int32),
                _squared_norm(recursive_residual),
                true_residual,
                true_error,
            )

        return lax.cond(
            start_is_valid,
            valid_start,
            invalid_start,
            operand=None,
        )

    (
        solver,
        solution,
        iteration,
        converged,
        breakdown,
        restarts,
        recursive_error,
        true_residual,
        true_error,
    ) = lax.while_loop(restart_cond, run_segment, initial)
    info = {
        # Keep the historical squared-norm meaning of ``error``, now made
        # reliable by basing it on the true residual.
        "error": true_error,
        "recursive_error": recursive_error,
        "relative_residual": _relative_residual(true_error, b_norm_squared),
        "converged": converged,
        "breakdown": breakdown,
        "n_iter": iteration,
        "n_restarts": restarts,
    }
    return solver, solution, info


def _cg_iterations(matvec, b, x0, precondition, tol, atol, maxiter):
    def stateful_precondition(unused_state, residual):
        return unused_state, precondition(residual)

    _, solution, info = _cg_iterations_stateful(
        (),
        matvec,
        b,
        x0,
        stateful_precondition,
        tol,
        atol,
        maxiter,
    )
    return solution, info


def cg_solve(K, f, x0=None, tol=1e-5, maxiter=None, M=None, *, atol=0.0):
    """Differentiable reliable preconditioned CG using public JAX APIs."""

    b = jnp.asarray(f)
    initial = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0)
    if initial.shape != b.shape:
        raise ValueError(
            f"x0 and f must have matching shapes: {initial.shape} vs {b.shape}"
        )
    if maxiter is None:
        maxiter = 10 * b.size
    if M is None:
        M = lambda value: value

    def solve(matvec, right_hand_side):
        return _cg_iterations(
            matvec,
            right_hand_side,
            initial,
            M,
            tol,
            atol,
            maxiter,
        )

    return lax.custom_linear_solve(
        K,
        b,
        solve=solve,
        symmetric=True,
        has_aux=True,
    )


def cg_solve_stateful(
    solver,
    matvec,
    right_hand_side,
    x0=None,
    *,
    precondition,
    tol=1e-5,
    atol=0.0,
    maxiter=None,
):
    """Reliable preconditioned CG with explicitly threaded solver state."""

    b = jnp.asarray(right_hand_side)
    if x0 is None:
        x0 = jnp.zeros_like(b)
    else:
        x0 = jnp.asarray(x0)
    if x0.shape != b.shape:
        raise ValueError(
            "x0 and right_hand_side must have matching shapes: "
            f"{x0.shape} vs {b.shape}"
        )
    if maxiter is None:
        maxiter = 10 * b.size

    return _cg_iterations_stateful(
        solver,
        matvec,
        b,
        x0,
        precondition,
        tol,
        atol,
        maxiter,
    )
