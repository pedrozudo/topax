import jax.numpy as jnp


PROBLEM_COMPLIANCE = "compliance"

FILTER_NONE = "none"
FILTER_HELMHOLTZ = "helmholtz"
FILTER_CONVOLUTION = "convolution"

FORMULATION_MMA = "mma"
# FORMULATION_AUGMENTED_LAGRANGIAN = "augmented_lagrangian"


_REGISTRY = {}


def volume_constraint(rho, volfrac):
    return jnp.mean(rho) - volfrac


def normalized_volume_constraint(rho, volfrac):
    return jnp.mean(rho) / volfrac - 1.0


def register_value_and_grad(
    problem,
    filtering,
    formulation=FORMULATION_MMA,
):
    key = (
        problem,
        filtering,
        formulation,
    )

    def decorator(fn):
        if key in _REGISTRY:
            raise ValueError(
                f"Duplicate value_and_grad registration for configuration {key}"
            )

        _REGISTRY[key] = fn

        return fn

    return decorator


def available_value_and_grad_constructors():
    return tuple(sorted(_REGISTRY.keys()))


def construct_value_and_grad(
    problem,
    filtering,
    *args,
    formulation=FORMULATION_MMA,
    constraint_function=volume_constraint,
    **kwargs,
):
    key = (
        problem,
        filtering,
        formulation,
    )

    try:
        builder = _REGISTRY[key]

    except KeyError as exc:
        available = ", ".join(
            map(
                str,
                available_value_and_grad_constructors(),
            )
        )

        raise ValueError(
            f"Unknown value_and_grad configuration {key}. "
            f"Available configurations: {available}"
        ) from exc

    kwargs["constraint_function"] = constraint_function

    return builder(
        *args,
        **kwargs,
    )
