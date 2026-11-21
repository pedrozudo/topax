from .elastic import (
    Elastic,
    construct_elastic,
    construct_local_stiffness,
    elastic_n_dofs_node,
)
from .helmholtz import (
    Helmholtz,
    construct_helmholtz,
    construct_local_helmholtz,
    helmholtz_n_dofs_node,
    helmholtz_to_nodal,
    helmholtz_to_elemental,
)

from .density import construct_conv_filter, construct_heaviside


def material_n_dofs_node(material, n_dims):
    if isinstance(material, Elastic):
        return elastic_n_dofs_node(n_dims)
    if isinstance(material, Helmholtz):
        return helmholtz_n_dofs_node(n_dims)
    raise TypeError(f"Unsupported material type: {type(material).__name__}")


__all__ = [
    "construct_elastic",
    "construct_helmholtz",
    "construct_conv_filter",
    "construct_heaviside",
    "construct_local_stiffness",
    "construct_local_helmholtz",
    "elastic_n_dofs_node",
    "helmholtz_n_dofs_node",
    "material_n_dofs_node",
    "helmholtz_to_nodal",
    "helmholtz_to_elemental",
]
