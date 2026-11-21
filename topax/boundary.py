import numpy as np
import jax.numpy as jnp

from .material import material_n_dofs_node


def construct_restrictions(grid, material, dirichlet_boundary_conditions=None):
    """Construct a Galerkin-compatible hierarchy of Dirichlet masks.

    The physical boundary condition is constructed exactly on the fine grid.
    Coarser restriction masks are then derived recursively so that every
    active coarse-grid basis function has prolongation support only on active
    fine-grid DOFs.
    """
    n_dofs_node = material_n_dofs_node(material, grid.n_dims)

    if dirichlet_boundary_conditions is None:
        dirichlet_boundary_conditions = [[] for _ in range(n_dofs_node)]

    if len(dirichlet_boundary_conditions) != n_dofs_node:
        raise ValueError(
            "Expected one list of Dirichlet boundary conditions "
            f"per DOF component ({n_dofs_node}), got "
            f"{len(dirichlet_boundary_conditions)}."
        )

    # Construct the physical Dirichlet condition exactly on the fine grid.
    fine_restriction = construct_level_restrictions(
        grid.levels[0],
        dirichlet_boundary_conditions,
    )

    restrictions = [fine_restriction]

    # Import locally to avoid a boundary/vcycle module import cycle.
    from .vcycle import restrict_residual

    # Construct Galerkin-safe coarse masks recursively.
    #
    # A coarse DOF must be restricted whenever its prolongation basis
    # function touches any restricted fine-grid DOF.
    for level_index in range(grid.n_levels - 1):
        fine_restriction = restrictions[-1]

        fine_blocked = (~fine_restriction).astype(jnp.float32)

        coarse_blocked = restrict_residual(
            fine_blocked,
            grid,
            restrictions=None,
            level_index=level_index,
        )

        coarse_restriction = coarse_blocked == 0

        restrictions.append(coarse_restriction)

    return tuple(restrictions)


def validate_galerkin_restrictions(grid, restrictions):
    """Reject masks that do not preserve the Galerkin prolongation space.

    Exact fixed-``G_s`` operators require every restricted fine node to be
    zero for every active coarse-grid basis function. Geometrically aligned
    faces satisfy this condition; arbitrary embedded or non-aligned masks may
    not. This validation must be performed during Galerkin solver setup,
    outside JAX transformations.
    """

    if len(restrictions) != grid.n_levels:
        raise ValueError("Expected one restriction mask per grid level.")

    restrictions = tuple(
        jnp.asarray(mask, dtype=jnp.bool_)
        for mask in restrictions
    )

    for level_index, (level, restriction) in enumerate(
        zip(grid.levels, restrictions)
    ):
        expected_spatial_shape = tuple(map(int, level.nodes))

        if restriction.ndim != grid.n_dims + 1:
            raise ValueError(
                "Restriction masks must have one component axis followed by "
                "the grid-node axes."
            )

        if restriction.shape[1:] != expected_spatial_shape:
            raise ValueError(
                f"Restriction mask {level_index} has spatial shape "
                f"{restriction.shape[1:]}, expected "
                f"{expected_spatial_shape}."
            )

        if (
            level_index > 0
            and restriction.shape[0] != restrictions[0].shape[0]
        ):
            raise ValueError(
                "Restriction masks must have the same component count on "
                "every level."
            )

    # Import locally to avoid a module-level boundary/vcycle import cycle.
    from .vcycle import prolongate_correction

    for level_index in range(grid.n_levels - 1):
        fine_restriction = restrictions[level_index]
        coarse_restriction = restrictions[level_index + 1]

        prolonged_active_support = prolongate_correction(
            coarse_restriction.astype(jnp.float32),
            grid,
            restrictions=None,
            level_index=level_index,
        )

        incompatible = (
            (~fine_restriction)
            & (prolonged_active_support > 0)
        )

        if bool(np.asarray(jnp.any(incompatible))):
            raise ValueError(
                "Dirichlet restrictions are incompatible with exact "
                "Galerkin coarsening between levels "
                f"{level_index} and {level_index + 1}; "
                "restricted fine nodes must not overlap the prolongation "
                "support of active coarse DOFs."
            )

    return restrictions


def construct_level_restrictions(
    level,
    dirichlet_boundary_conditions,
):
    restrictions = []

    for component_boundary_conditions in dirichlet_boundary_conditions:
        component_restrictions = jnp.ones(
            tuple(level.nodes),
            dtype=jnp.bool_,
        )

        for boundary_condition in component_boundary_conditions:
            slices = compute_slices(
                level,
                boundary_condition["center"],
                boundary_condition["size"],
            )

            component_restrictions = component_restrictions.at[
                slices
            ].set(False)

        restrictions.append(component_restrictions)

    return jnp.stack(restrictions, axis=0)


def compute_slices(level, center, size):
    center = np.asarray(center) * level.elements
    size = np.asarray(size) * level.elements

    lower_index = np.maximum(
        0,
        np.round(center - size / 2).astype(int),
    )

    upper_index = np.minimum(
        np.round(center + size / 2).astype(int) + 1,
        level.nodes,
    )

    return tuple(
        slice(
            int(lo),
            int(up if up > lo else lo + 1),
        )
        for lo, up in zip(lower_index, upper_index)
    )