import jax.numpy as jnp
import flax


@flax.struct.dataclass
class ValueAndGrad:
    value: jnp.ndarray
    grad: jnp.ndarray
