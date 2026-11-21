"""Method of moving asymptotes optimizer."""

import functools

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class MMA1ConParams:
    xmin = 0.0
    xmax = 1.0

    move: float = 0.2
    asyinit: float = 0.5
    asyincr: float = 1.2
    asydecr: float = 0.7
    albefa: float = 0.1
    asym_min_ratio: float = 0.01
    asym_max_ratio: float = 10.0

    # scaling
    # https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.3321
    objective_scale: float = 100.0

    # MMA stabilizers
    raa0: float = 1e-5

    # bisection settings
    max_bisect: int = 50
    lam_max: float = 1e8

    # numerics
    eps: float = 1e-12


@flax.struct.dataclass
class MMA1ConState:
    # all shape (n,)
    xold1: jnp.ndarray
    xold2: jnp.ndarray
    low: jnp.ndarray
    upp: jnp.ndarray
    it: jnp.ndarray  # int scalar


@flax.struct.dataclass
class MMAApprox:
    # asymptotes and bounds (shape (n,))
    low: jnp.ndarray
    upp: jnp.ndarray
    alpha: jnp.ndarray
    beta: jnp.ndarray

    # separable coefficients (shape (n,))
    p0: jnp.ndarray
    q0: jnp.ndarray
    P: jnp.ndarray
    Q: jnp.ndarray

    # scalar rhs such that phi(0)=g (scalar)
    b: jnp.ndarray


def init_mma_1con_params(**kwargs):
    params = MMA1ConParams(**kwargs)
    return params


def init_mma_1con_state(x, mma_params):
    x = x.reshape(-1)

    range_x = mma_params.xmax - mma_params.xmin
    low = jnp.maximum(
        x - mma_params.asyinit * range_x,
        mma_params.xmin,
    )
    upp = jnp.minimum(
        x + mma_params.asyinit * range_x,
        mma_params.xmax,
    )

    state = MMA1ConState(
        xold1=x,
        xold2=x,
        low=low,
        upp=upp,
        it=jnp.array(1, dtype=jnp.int32),
    )
    return state


def _clamp(x, lo, hi):
    return jnp.minimum(jnp.maximum(x, lo), hi)


def _mma_update_asymptotes(
    x,
    state: MMA1ConState,
    params: MMA1ConParams,
):
    x_range = jnp.maximum(
        params.xmax - params.xmin,
        params.eps,
    )

    def asym_init(_):
        low = x - params.asyinit * x_range
        upp = x + params.asyinit * x_range
        return low, upp

    def asym_update(_):
        # zzz = (x^k - x^{k-1}) * (x^{k-1} - x^{k-2})
        zzz = (x - state.xold1) * (state.xold1 - state.xold2)

        factor = jnp.where(
            zzz > 0.0,
            params.asyincr,
            jnp.where(
                zzz < 0.0,
                params.asydecr,
                1.0,
            ),
        )

        low = x - factor * (state.xold1 - state.low)

        upp = x + factor * (state.upp - state.xold1)

        # Clamp asymptotes.
        low = _clamp(
            low,
            x - params.asym_max_ratio * x_range,
            x - params.asym_min_ratio * x_range,
        )

        upp = _clamp(
            upp,
            x + params.asym_min_ratio * x_range,
            x + params.asym_max_ratio * x_range,
        )

        return low, upp

    return jax.lax.cond(
        state.it <= 2,
        asym_init,
        asym_update,
        operand=None,
    )


def _mma_build_approx_1con(
    *,
    x,
    df0dx,
    g,
    dgdx,
    low,
    upp,
    params: MMA1ConParams,
) -> MMAApprox:
    x_range = jnp.maximum(
        params.xmax - params.xmin,
        params.eps,
    )

    # Alpha/beta:
    # design bounds + move limit + albefa.
    alpha = jnp.maximum(
        low + params.albefa * (x - low),
        x - params.move * x_range,
    )
    alpha = jnp.maximum(
        alpha,
        params.xmin,
    )

    beta = jnp.minimum(
        upp - params.albefa * (upp - x),
        x + params.move * x_range,
    )
    beta = jnp.minimum(
        beta,
        params.xmax,
    )

    # Distances to asymptotes.
    ux = jnp.maximum(
        upp - x,
        params.eps,
    )
    xl = jnp.maximum(
        x - low,
        params.eps,
    )

    ux2 = ux * ux
    xl2 = xl * xl

    inv_xmami = 1.0 / jnp.maximum(
        x_range,
        1e-5,
    )

    # Objective coefficients.
    p0 = jnp.maximum(
        df0dx,
        0.0,
    )
    q0 = jnp.maximum(
        -df0dx,
        0.0,
    )

    pq0 = 0.001 * (p0 + q0) + params.raa0 * inv_xmami

    p0 = (p0 + pq0) * ux2

    q0 = (q0 + pq0) * xl2

    # Constraint coefficients.
    # Single constraint -> vectors.
    P = jnp.maximum(
        dgdx,
        0.0,
    )
    Q = jnp.maximum(
        -dgdx,
        0.0,
    )

    PQ = 0.001 * (P + Q) + params.raa0 * inv_xmami

    P = (P + PQ) * ux2

    Q = (Q + PQ) * xl2

    # Choose b so phi(0) == g.
    b = jnp.sum(P / ux + Q / xl) - g

    return MMAApprox(
        low=low,
        upp=upp,
        alpha=alpha,
        beta=beta,
        p0=p0,
        q0=q0,
        P=P,
        Q=Q,
        b=b,
    )


def _mma_subsolve_1con_bisect(
    approx: MMAApprox,
    params: MMA1ConParams,
):
    """
    Solve the one-constraint MMA subproblem by finding lambda >= 0
    such that phi(lambda) = 0.

    Returns
    -------
    xnew
        Updated design variables.

    lam_star
        Optimal dual variable.

    phi_star
        Constraint residual of the MMA approximation at lam_star.

    bracketed
        Whether a feasible upper bound for the dual root was found.
    """
    eps = params.eps

    low = approx.low
    upp = approx.upp
    alpha = approx.alpha
    beta = approx.beta

    p0 = approx.p0
    q0 = approx.q0
    P = approx.P
    Q = approx.Q
    b = approx.b

    dtype = low.dtype

    def x_of(lam):
        A = jnp.maximum(
            p0 + lam * P,
            eps,
        )

        B = jnp.maximum(
            q0 + lam * Q,
            eps,
        )

        sqrt_A = jnp.sqrt(A)
        sqrt_B = jnp.sqrt(B)

        xhat = (sqrt_A * low + sqrt_B * upp) / (sqrt_A + sqrt_B)

        return _clamp(
            xhat,
            alpha,
            beta,
        )

    def phi(lam):
        xlam = x_of(lam)

        ux = jnp.maximum(
            upp - xlam,
            eps,
        )

        xl = jnp.maximum(
            xlam - low,
            eps,
        )

        return jnp.sum(P / ux + Q / xl) - b

    lam_zero = jnp.array(
        0.0,
        dtype=dtype,
    )

    phi_zero = phi(lam_zero)

    def inactive(_):
        xnew = x_of(lam_zero)

        return (
            xnew,
            lam_zero,
            phi_zero,
            jnp.asarray(True),
        )

    def active(_):
        # -------------------------------------------------------------
        # Find an upper bound lambda_hi such that phi(lambda_hi) <= 0.
        # -------------------------------------------------------------
        lam_hi = jnp.array(
            1.0,
            dtype=dtype,
        )

        phi_hi = phi(lam_hi)

        def bracket_cond(carry):
            lam_hi, phi_hi = carry

            return jnp.logical_and(
                phi_hi > 0.0,
                lam_hi < params.lam_max,
            )

        def bracket_body(carry):
            lam_hi, _ = carry

            lam_hi = jnp.minimum(
                2.0 * lam_hi,
                params.lam_max,
            )

            phi_hi = phi(lam_hi)

            return (
                lam_hi,
                phi_hi,
            )

        lam_hi, phi_hi = jax.lax.while_loop(
            bracket_cond,
            bracket_body,
            (
                lam_hi,
                phi_hi,
            ),
        )

        bracketed = phi_hi <= 0.0

        lam_lo = lam_zero

        # -------------------------------------------------------------
        # Fixed-iteration bisection.
        #
        # We deliberately use a fixed number of iterations here.
        # The previous phi-based stopping condition could terminate
        # using phi from one midpoint and then return the midpoint of
        # the subsequently updated bracket.
        # -------------------------------------------------------------
        def bisect_body(
            _,
            bracket,
        ):
            lam_lo, lam_hi = bracket

            lam_mid = 0.5 * (lam_lo + lam_hi)

            phi_mid = phi(lam_mid)

            # phi(lambda) decreases with lambda.
            #
            # phi_mid > 0:
            #     lambda needs to increase.
            #
            # phi_mid <= 0:
            #     lambda can decrease.
            lam_lo = jnp.where(
                phi_mid > 0.0,
                lam_mid,
                lam_lo,
            )

            lam_hi = jnp.where(
                phi_mid > 0.0,
                lam_hi,
                lam_mid,
            )

            return (
                lam_lo,
                lam_hi,
            )

        lam_lo, lam_hi = jax.lax.fori_loop(
            0,
            params.max_bisect,
            bisect_body,
            (
                lam_lo,
                lam_hi,
            ),
        )

        # The upper side is feasible when a bracket exists. Otherwise return
        # the limiting candidate and expose the failed bracket to the caller.
        lam_star = jnp.where(bracketed, lam_hi, params.lam_max)

        xnew = x_of(lam_star)
        phi_star = phi(lam_star)

        return (
            xnew,
            lam_star,
            phi_star,
            bracketed,
        )

    return jax.lax.cond(
        phi_zero <= 0.0,
        inactive,
        active,
        operand=None,
    )


def _mma_step_1con_bisect(
    *,
    x,
    df0dx,
    g,
    dgdx,
    state: MMA1ConState,
    params: MMA1ConParams,
):
    low, upp = _mma_update_asymptotes(
        x,
        state,
        params,
    )

    approx = _mma_build_approx_1con(
        x=x,
        df0dx=df0dx,
        g=g,
        dgdx=dgdx,
        low=low,
        upp=upp,
        params=params,
    )

    xnew, lam_star, phi_star, bracketed = _mma_subsolve_1con_bisect(
        approx,
        params,
    )

    state_new = state.replace(
        xold2=state.xold1,
        xold1=x,
        low=low,
        upp=upp,
        it=(
            state.it
            + jnp.array(
                1,
                dtype=state.it.dtype,
            )
        ),
    )

    info = {
        "lam": lam_star,
        "phi": phi_star,
        "bracketed": bracketed,
        # Bisection can leave a positive residual at round-off scale even
        # though a valid bracket was found (for example, +2e-16 in float64).
        "feasible": bracketed & (phi_star <= params.eps),
    }

    return (
        xnew,
        state_new,
        info,
    )


def mma_step(
    x,
    o_value,
    o_grad,
    c_value,
    c_grad,
    state: MMA1ConState,
    *,
    params: MMA1ConParams,
):
    x_shape = x.shape

    # Objective scaling:
    # scale gradient by objective_scale / max(|f|, eps).
    scale = params.objective_scale / jnp.maximum(
        jnp.abs(o_value),
        1e-12,
    )

    df0dx = (scale * o_grad).reshape(-1)

    x = x.reshape(-1)

    g = c_value
    dgdx = c_grad.reshape(-1)

    x_new, state_new, info = _mma_step_1con_bisect(
        x=x,
        df0dx=df0dx,
        g=g,
        dgdx=dgdx,
        state=state,
        params=params,
    )

    x_new = x_new.reshape(x_shape)

    return (
        x_new,
        state_new,
        info,
    )


def construct_mma_step(
    mma_parameters: MMA1ConParams,
):
    return functools.partial(
        mma_step,
        params=mma_parameters,
    )
