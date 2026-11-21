import numbers

import jax


def jacobi_smooth(solution, residual, inverse_diagonal, damping):
    return solution + damping * inverse_diagonal * residual


def jacobi_smooth_n(
    solution,
    right_hand_side,
    inverse_diagonal,
    damping,
    apply_operator,
    n_smoothing_steps,
    *,
    initial_residual=None,
):
    """Apply one or more damped-Jacobi smoothing steps.

    An initial residual avoids an unnecessary operator application when the
    caller already knows the residual. In particular, a V-cycle starts its
    pre-smoothing from zero, where the residual is the right-hand side.
    """

    if isinstance(n_smoothing_steps, (bool, jax.Array)) or not isinstance(
        n_smoothing_steps,
        numbers.Integral,
    ):
        raise ValueError("n_smoothing_steps must be a positive integer.")
    if n_smoothing_steps < 1:
        raise ValueError("n_smoothing_steps must be a positive integer.")

    first_step = 0
    if initial_residual is not None:
        solution = jacobi_smooth(
            solution,
            initial_residual,
            inverse_diagonal,
            damping,
        )
        first_step = 1

    def body_fun(_, current_solution):
        residual = right_hand_side - apply_operator(current_solution)
        return jacobi_smooth(
            current_solution,
            residual,
            inverse_diagonal,
            damping,
        )

    return jax.lax.fori_loop(
        first_step,
        int(n_smoothing_steps),
        body_fun,
        solution,
    )
