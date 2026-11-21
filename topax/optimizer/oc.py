"""Optimality-criteria optimizer."""

import functools

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class OCParams:
    xmin: float = 0.0
    xmax: float = 1.0
    move: float = 0.2
    damping: float = 0.5

    max_bisect: int = 64
    bisect_tol: float = 1.0e-6
    lam_min: float = 0.0
    lam_max: float = 1.0e9

    eps: float = 1.0e-30


def init_oc_params(**kwargs):
    return OCParams(**kwargs)


def oc_step(
    x,
    objective_grad,
    constraint_value,
    constraint_grad,
    *,
    params: OCParams,
):
    """Perform one bounded optimality-criteria design update.

    The constraint is linearized as

        constraint_value + constraint_grad @ (x_new - x) <= 0.

    Standard density-based OC assumes a non-positive objective gradient and
    a positive constraint gradient.
    """

    dtype = x.dtype
    constraint_grad = jnp.maximum(
        constraint_grad,
        jnp.asarray(params.eps, dtype=dtype),
    )

    lower = jnp.maximum(params.xmin, x - params.move)
    upper = jnp.minimum(params.xmax, x + params.move)

    def candidate(lam):
        ratio = -objective_grad / (constraint_grad * lam)
        ratio = jnp.maximum(ratio, 0.0)
        unconstrained = x * ratio**params.damping
        return jnp.clip(unconstrained, lower, upper)

    def trial_constraint(x_trial):
        return constraint_value + jnp.sum(
            constraint_grad * (x_trial - x)
        )

    def body(_, bracket):
        lam_lower, lam_upper = bracket
        relative_width = (lam_upper - lam_lower) / (
            lam_lower + lam_upper
        )
        active = jnp.logical_or(
            params.bisect_tol <= 0.0,
            relative_width > params.bisect_tol,
        )

        lam_mid = 0.5 * (lam_lower + lam_upper)
        constraint_mid = trial_constraint(candidate(lam_mid))
        increase_lambda = constraint_mid > 0.0

        next_lower = jnp.where(
            increase_lambda,
            lam_mid,
            lam_lower,
        )
        next_upper = jnp.where(
            increase_lambda,
            lam_upper,
            lam_mid,
        )
        return (
            jnp.where(active, next_lower, lam_lower),
            jnp.where(active, next_upper, lam_upper),
        )

    bracket = jax.lax.fori_loop(
        0,
        params.max_bisect,
        body,
        (
            jnp.asarray(params.lam_min, dtype=dtype),
            jnp.asarray(params.lam_max, dtype=dtype),
        ),
    )
    lam = 0.5 * (bracket[0] + bracket[1])
    x_new = candidate(lam)
    info = {
        "change": jnp.max(jnp.abs(x_new - x)),
        "constraint": trial_constraint(x_new),
        "lam": lam,
    }
    return x_new, info


def construct_oc_step(oc_parameters: OCParams):
    return functools.partial(
        oc_step,
        params=oc_parameters,
    )
