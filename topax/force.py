import jax.numpy as jnp

from .boundary import compute_slices
from .grid import Grid
from .material import material_n_dofs_node


def construct_forces(grid: Grid, physics, concentrated_forces):
    n_dofs_node = material_n_dofs_node(physics, grid.n_dims)

    if len(concentrated_forces) != n_dofs_node:
        raise ValueError(
            "Expected one list of forces "
            f"per DOF component ({n_dofs_node}), got "
            f"{len(concentrated_forces)}."
        )

    level = grid.fine_level
    dof_forces = []

    for component_forces in concentrated_forces:
        forces = jnp.zeros(tuple(level.nodes))
        for region_force in component_forces:
            slices = compute_slices(level, region_force["center"], region_force["size"])
            forces = forces.at[slices].set(region_force["intensity"])
        dof_forces.append(forces)

    return jnp.stack(dof_forces, axis=0)
