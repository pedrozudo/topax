# Expose submodules for structured access to internal functionality
from . import grid
from . import boundary
from . import material
from . import vcycle
from . import sparse
from . import solver
from . import operator
from . import value_grad
from . import optimizer
from . import cg
from . import smoother
from . import galerkin

# High-level value-and-gradient interface
from .material import (
    construct_helmholtz,
    construct_elastic,
    construct_local_stiffness,
    construct_local_helmholtz,
    helmholtz_to_nodal,
    helmholtz_to_elemental,
    construct_conv_filter,
    construct_heaviside,
)
from .grid import construct_grid
from .boundary import (
    construct_restrictions,
    validate_galerkin_restrictions,
)
from .force import construct_forces
from .smoother import jacobi_smooth, jacobi_smooth_n

from .value_grad import (
    construct_value_and_grad,
    available_value_and_grad_constructors,
    volume_constraint,
    normalized_volume_constraint,
    ComplianceSolveDiagnostics,
    PROBLEM_COMPLIANCE,
    FILTER_NONE,
    FILTER_HELMHOLTZ,
    FILTER_CONVOLUTION,
    FORMULATION_MMA,
)


__all__ = [
    # Submodules
    "grid",
    "boundary",
    "material",
    "vcycle",
    "sparse",
    "solver",
    "operator",
    "value_grad",
    "optimizer",
    "cg",
    "smoother",
    "galerkin",
    # Value-and-grad interface
    "construct_grid",
    "construct_restrictions",
    "validate_galerkin_restrictions",
    "construct_helmholtz",
    "construct_conv_filter",
    "construct_heaviside",
    "construct_elastic",
    "construct_local_stiffness",
    "construct_local_helmholtz",
    "helmholtz_to_nodal",
    "helmholtz_to_elemental",
    "construct_forces",
    "jacobi_smooth",
    "jacobi_smooth_n",
    "construct_value_and_grad",
    "available_value_and_grad_constructors",
    "volume_constraint",
    "normalized_volume_constraint",
    "ComplianceSolveDiagnostics",
    # MODES
    "FILTER_NONE",
    "PROBLEM_COMPLIANCE",
    "FILTER_HELMHOLTZ",
    "FILTER_CONVOLUTION",
    "FORMULATION_MMA",
]
