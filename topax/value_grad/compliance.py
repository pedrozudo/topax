import functools

import flax
import jax
import jax.numpy as jnp

from ..grid import Grid
from ..operator import (
    DEFAULT_SLAB_SIZE,
    element_strain_energy,
)
from ..vcycle import (
    helmholtz_vcycle_solve,
    physics_vcycle_solve,
)
from ..smoother import jacobi_smooth_n
from ..material import helmholtz_to_nodal, helmholtz_to_elemental

from .types import ValueAndGrad

from .dispatcher import (
    FILTER_NONE,
    FILTER_HELMHOLTZ,
    FILTER_CONVOLUTION,
    FORMULATION_MMA,
    PROBLEM_COMPLIANCE,
    register_value_and_grad,
)


@flax.struct.dataclass
class CGStateComplianceNoFiltering:
    physics_value0: jnp.ndarray


@flax.struct.dataclass
class CGStateComplianceConvFiltering:
    physics_value0: jnp.ndarray


@flax.struct.dataclass
class CGStateComplianceCGFiltering:
    helmholtz_value0: jnp.ndarray
    helmholtz_constraint_grad0: jnp.ndarray
    helmholtz_objective_grad0: jnp.ndarray
    physics_value0: jnp.ndarray


@flax.struct.dataclass
class ComplianceSolveDiagnostics:
    physics: dict
    helmholtz_forward: dict | None = None
    helmholtz_objective_adjoint: dict | None = None
    helmholtz_constraint_adjoint: dict | None = None

    @property
    def converged(self):
        infos = (
            self.physics,
            self.helmholtz_forward,
            self.helmholtz_objective_adjoint,
            self.helmholtz_constraint_adjoint,
        )
        flags = [info["converged"] for info in infos if info is not None]
        return jnp.all(jnp.stack(flags))


def n_dofs_node_from_local_operator(
    level,
    local_operator,
):
    n_dofs_element = local_operator.shape[0]

    if n_dofs_element % level.n_nodes_element != 0:
        raise ValueError("Local operator is incompatible with the element topology.")

    return n_dofs_element // level.n_nodes_element


def _compliance_value_and_grad(
    level,
    displacement,
    forces,
    local_operator,
    rho,
    simp_penalty,
    rho_min,
    restrictions,
):
    """Evaluate compliance and its elemental SIMP derivative directly."""

    strain_energy = element_strain_energy(
        level,
        displacement,
        local_operator,
        restrictions,
    )

    objective_value = jnp.vdot(forces, displacement).real
    stiffness_derivative = (
        simp_penalty
        * (1.0 - rho_min)
        * rho ** (simp_penalty - 1.0)
        * strain_energy
    )
    # Compliance sensitivity for an equilibrium displacement:
    #
    #     dC / drho = -u^T (dK / drho) u
    return objective_value, -stiffness_derivative


def init_cg_state_no_filtering(
    elements,
    n_dofs_node,
):
    nodal_shape = tuple(int(n) + 1 for n in elements)

    physics_value0 = jnp.zeros(
        (
            n_dofs_node,
            *nodal_shape,
        )
    )

    return CGStateComplianceNoFiltering(physics_value0=physics_value0)


def init_cg_state_conv_filtering(
    elements,
    n_dofs_node,
):
    nodal_shape = tuple(int(n) + 1 for n in elements)

    physics_value0 = jnp.zeros(
        (
            n_dofs_node,
            *nodal_shape,
        )
    )

    return CGStateComplianceConvFiltering(physics_value0=physics_value0)


def init_cg_state_cg_filtering(
    elements,
    n_dofs_node,
):
    nodal_shape = tuple(int(n) + 1 for n in elements)

    physics_value0 = jnp.zeros(
        (
            n_dofs_node,
            *nodal_shape,
        )
    )

    helmholtz_value0 = jnp.zeros(
        (
            1,
            *nodal_shape,
        )
    )

    return CGStateComplianceCGFiltering(
        helmholtz_value0=helmholtz_value0,
        helmholtz_constraint_grad0=helmholtz_value0,
        helmholtz_objective_grad0=helmholtz_value0,
        physics_value0=physics_value0,
    )


@register_value_and_grad(
    PROBLEM_COMPLIANCE,
    FILTER_NONE,
    FORMULATION_MMA,
)
def construct_value_and_grad_compliance_no_filtering(
    fine_local_operator,
    grid_physics: Grid,
    coarse_physics_structure,
    simp_penalty,
    rho_min,
    tol,
    maxiter,
    damping,
    volfrac,
    constraint_function,
    n_smoothing_steps=1,
    smoother_impl=jacobi_smooth_n,
    galerkin_crossover_level=None,
    slab_size=DEFAULT_SLAB_SIZE,
    slab_axis=None,
):
    solve_physics = functools.partial(
        physics_vcycle_solve,
        grid=grid_physics,
        fine_local_operator=fine_local_operator,
        coarse_structure=coarse_physics_structure,
        galerkin_crossover_level=galerkin_crossover_level,
        simp_penalty=simp_penalty,
        rho_min=rho_min,
        damping=damping,
        tol=tol,
        maxiter=maxiter,
        n_smoothing_steps=n_smoothing_steps,
        smoother_impl=smoother_impl,
        slab_size=slab_size,
        slab_axis=slab_axis,
    )

    def value_and_grad(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        state: CGStateComplianceNoFiltering,
    ):
        rho_physical = rho

        physics_solver, u, physics_cg_info = solve_physics(
            solver=solvers.physics,
            rho=rho_physical,
            forces=forces,
            restrictions=restrictions,
            x0=state.physics_value0,
        )

        def _constraint(rho):
            return constraint_function(
                rho,
                volfrac,
            )

        (
            objective_val,
            objective_grad,
        ) = _compliance_value_and_grad(
            grid_physics.fine_level,
            u,
            forces,
            fine_local_operator,
            rho_physical,
            simp_penalty,
            rho_min,
            restrictions[0],
        )

        (
            constraint_val,
            constraint_grad,
        ) = jax.value_and_grad(
            _constraint,
            argnums=0,
        )(rho_physical)

        objective = ValueAndGrad(
            value=objective_val,
            grad=objective_grad,
        )

        constraint = ValueAndGrad(
            value=constraint_val,
            grad=constraint_grad,
        )

        solvers = solvers._replace(physics=physics_solver)
        state = state.replace(physics_value0=u)
        diagnostics = ComplianceSolveDiagnostics(physics=physics_cg_info)

        return (
            solvers,
            objective,
            constraint,
            state,
            diagnostics,
        )

    n_dofs_node = n_dofs_node_from_local_operator(
        grid_physics.fine_level,
        fine_local_operator,
    )

    initial_cg_state = init_cg_state_no_filtering(
        grid_physics.elements,
        n_dofs_node,
    )

    return (
        value_and_grad,
        initial_cg_state,
    )


@register_value_and_grad(
    PROBLEM_COMPLIANCE,
    FILTER_CONVOLUTION,
    FORMULATION_MMA,
)
def construct_value_and_grad_compliance_conv_filtering(
    fine_local_operator,
    grid_physics: Grid,
    coarse_physics_structure,
    conv_filter,
    projection,
    simp_penalty,
    rho_min,
    tol,
    maxiter,
    damping,
    volfrac,
    constraint_function,
    n_smoothing_steps=1,
    smoother_impl=jacobi_smooth_n,
    galerkin_crossover_level=None,
    slab_size=DEFAULT_SLAB_SIZE,
    slab_axis=None,
):
    solve_physics = functools.partial(
        physics_vcycle_solve,
        grid=grid_physics,
        fine_local_operator=fine_local_operator,
        coarse_structure=coarse_physics_structure,
        galerkin_crossover_level=galerkin_crossover_level,
        simp_penalty=simp_penalty,
        rho_min=rho_min,
        damping=damping,
        tol=tol,
        maxiter=maxiter,
        n_smoothing_steps=n_smoothing_steps,
        smoother_impl=smoother_impl,
        slab_size=slab_size,
        slab_axis=slab_axis,
    )

    def value_and_grad(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        state: CGStateComplianceConvFiltering,
    ):
        def _project(rho):
            rho_filtered = conv_filter(
                rho,
                rho_mask,
            )

            return projection(rho_filtered)

        def _constraint(
            rho_projected,
        ):
            return constraint_function(
                rho_projected,
                volfrac,
            )

        (
            rho_projected,
            pullback_project,
        ) = jax.vjp(
            _project,
            rho,
        )

        physics_solver, u, physics_cg_info = solve_physics(
            solver=solvers.physics,
            rho=rho_projected,
            forces=forces,
            restrictions=restrictions,
            x0=state.physics_value0,
        )

        (
            objective_val,
            objective_grad_projected,
        ) = _compliance_value_and_grad(
            grid_physics.fine_level,
            u,
            forces,
            fine_local_operator,
            rho_projected,
            simp_penalty,
            rho_min,
            restrictions[0],
        )
        (objective_grad,) = pullback_project(objective_grad_projected)

        (
            constraint_val,
            constraint_grad_projected,
        ) = jax.value_and_grad(_constraint)(rho_projected)

        (constraint_grad,) = pullback_project(constraint_grad_projected)

        objective = ValueAndGrad(
            value=objective_val,
            grad=objective_grad,
        )

        constraint = ValueAndGrad(
            value=constraint_val,
            grad=constraint_grad,
        )

        solvers = solvers._replace(physics=physics_solver)
        state = state.replace(physics_value0=u)
        diagnostics = ComplianceSolveDiagnostics(physics=physics_cg_info)

        return (
            solvers,
            objective,
            constraint,
            state,
            diagnostics,
        )

    n_dofs_node = n_dofs_node_from_local_operator(
        grid_physics.fine_level,
        fine_local_operator,
    )

    initial_cg_state = init_cg_state_conv_filtering(
        grid_physics.elements,
        n_dofs_node,
    )

    return (
        value_and_grad,
        initial_cg_state,
    )


@register_value_and_grad(
    PROBLEM_COMPLIANCE,
    FILTER_HELMHOLTZ,
    FORMULATION_MMA,
)
def construct_value_and_grad_compliance_cg_filtering(
    helmholtz,
    local_operators_helmholtz,
    fine_local_operator,
    grid: Grid,
    coarse_physics_structure,
    coarse_helmholtz_structure,
    projection,
    simp_penalty,
    rho_min,
    tol,
    maxiter,
    damping,
    volfrac,
    constraint_function,
    n_smoothing_steps=1,
    smoother_impl=jacobi_smooth_n,
    galerkin_crossover_level=None,
    slab_size=DEFAULT_SLAB_SIZE,
    slab_axis=None,
):
    # =========================================================================
    # Helmholtz V-cycle
    # =========================================================================

    solve_helmholtz = functools.partial(
        helmholtz_vcycle_solve,
        grid=grid,
        local_operators=local_operators_helmholtz,
        coarse_structure=coarse_helmholtz_structure,
        damping=damping,
        tol=tol,
        maxiter=maxiter,
        n_smoothing_steps=n_smoothing_steps,
        smoother_impl=smoother_impl,
        slab_size=slab_size,
        slab_axis=slab_axis,
    )

    # =========================================================================
    # Physics V-cycle
    # =========================================================================

    solve_physics = functools.partial(
        physics_vcycle_solve,
        grid=grid,
        fine_local_operator=fine_local_operator,
        coarse_structure=coarse_physics_structure,
        galerkin_crossover_level=galerkin_crossover_level,
        simp_penalty=simp_penalty,
        rho_min=rho_min,
        damping=damping,
        tol=tol,
        n_smoothing_steps=n_smoothing_steps,
        smoother_impl=smoother_impl,
        maxiter=maxiter,
        slab_size=slab_size,
        slab_axis=slab_axis,
    )

    fine_level = grid.fine_level

    # =========================================================================
    # Value + gradient
    # =========================================================================

    def value_and_grad(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        state: CGStateComplianceCGFiltering,
    ):
        # ---------------------------------------------------------------------
        # Elemental -> nodal Helmholtz RHS
        # ---------------------------------------------------------------------

        def _to_nodal(rho):
            return helmholtz_to_nodal(
                fine_level,
                helmholtz,
                rho,
            )

        rho_nodal, pullback_to_nodal = jax.vjp(
            _to_nodal,
            rho,
        )

        # ---------------------------------------------------------------------
        # Helmholtz filtering
        # ---------------------------------------------------------------------

        (
            helmholtz_solver,
            rho_nodal_filtered,
            helmholtz_forward_cg_info,
        ) = solve_helmholtz(
            solver=solvers.helmholtz,
            rho_nodal=rho_nodal,
            rho_mask=rho_mask,
            x0=state.helmholtz_value0,
        )

        # ---------------------------------------------------------------------
        # Nodal -> elemental + mask + projection
        # ---------------------------------------------------------------------

        def _project(
            rho_nodal_filtered,
        ):
            rho_elemental_filtered = helmholtz_to_elemental(
                fine_level,
                rho_nodal_filtered,
            )

            rho_elemental_filtered = rho_elemental_filtered * rho_mask

            return projection(rho_elemental_filtered)

        (
            rho_projected,
            pullback_project,
        ) = jax.vjp(
            _project,
            rho_nodal_filtered,
        )

        # ---------------------------------------------------------------------
        # Constraint
        # ---------------------------------------------------------------------

        def _constraint(
            rho_projected,
        ):
            return constraint_function(
                rho_projected,
                volfrac,
            )

        # ---------------------------------------------------------------------
        # Physics solve
        # ---------------------------------------------------------------------

        physics_solver, u, physics_cg_info = solve_physics(
            solver=solvers.physics,
            rho=rho_projected,
            forces=forces,
            restrictions=restrictions,
            x0=state.physics_value0,
        )

        # ---------------------------------------------------------------------
        # Compliance
        # ---------------------------------------------------------------------

        (
            objective_val,
            objective_grad_projected,
        ) = _compliance_value_and_grad(
            fine_level,
            u,
            forces,
            fine_local_operator,
            rho_projected,
            simp_penalty,
            rho_min,
            restrictions[0],
        )

        (
            constraint_val,
            constraint_grad_projected,
        ) = jax.value_and_grad(_constraint)(rho_projected)

        # ---------------------------------------------------------------------
        # Backpropagation through Helmholtz filter
        # ---------------------------------------------------------------------

        def _backprop_through_filter(
            grad_projected,
            helmholtz_grad0,
            helmholtz_solver,
        ):
            (grad_filtered,) = pullback_project(grad_projected)

            (
                helmholtz_solver,
                grad_nodal,
                cg_info,
            ) = solve_helmholtz(
                solver=helmholtz_solver,
                rho_nodal=grad_filtered,
                rho_mask=rho_mask,
                x0=helmholtz_grad0,
            )

            (grad_rho,) = pullback_to_nodal(grad_nodal)

            return (
                helmholtz_solver,
                grad_rho,
                grad_nodal,
                cg_info,
            )

        (
            helmholtz_solver,
            objective_grad,
            objective_grad_nodal,
            helmholtz_objective_cg_info,
        ) = _backprop_through_filter(
            objective_grad_projected,
            state.helmholtz_objective_grad0,
            helmholtz_solver,
        )

        (
            helmholtz_solver,
            constraint_grad,
            constraint_grad_nodal,
            helmholtz_constraint_cg_info,
        ) = _backprop_through_filter(
            constraint_grad_projected,
            state.helmholtz_constraint_grad0,
            helmholtz_solver,
        )

        # ---------------------------------------------------------------------
        # Results
        # ---------------------------------------------------------------------

        objective = ValueAndGrad(
            value=objective_val,
            grad=objective_grad,
        )

        constraint = ValueAndGrad(
            value=constraint_val,
            grad=constraint_grad,
        )

        solvers = solvers._replace(
            physics=physics_solver,
            helmholtz=helmholtz_solver,
        )
        state = state.replace(
            helmholtz_value0=rho_nodal_filtered,
            helmholtz_objective_grad0=objective_grad_nodal,
            helmholtz_constraint_grad0=constraint_grad_nodal,
            physics_value0=u,
        )
        diagnostics = ComplianceSolveDiagnostics(
            physics=physics_cg_info,
            helmholtz_forward=helmholtz_forward_cg_info,
            helmholtz_objective_adjoint=helmholtz_objective_cg_info,
            helmholtz_constraint_adjoint=helmholtz_constraint_cg_info,
        )

        return (
            solvers,
            objective,
            constraint,
            state,
            diagnostics,
        )

    # =========================================================================
    # Initial CG state
    # =========================================================================

    n_dofs_node = n_dofs_node_from_local_operator(
        fine_level,
        fine_local_operator,
    )

    initial_cg_state = init_cg_state_cg_filtering(
        grid.elements,
        n_dofs_node,
    )

    return (
        value_and_grad,
        initial_cg_state,
    )
