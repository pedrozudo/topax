"""Wang et al. (2025) large-scale 3D cantilever benchmark.

Reproduces the Fig. 10 / Table 1 topology-optimization problem from:

    J. Wang, N. Aage, J. Wu, O. Sigmund, R. Westermann,
    "Efficient large-scale 3D topology optimization with matrix-free
    MATLAB code", Structural and Multidisciplinary Optimization 68
    (2025), article 174.

The optimization follows the authors' released TOP3D_XL implementation:
the design density is used directly for elasticity, while the PDE operator
filters the compliance sensitivity as

    filtered_dc = PDE(rho * dc) / max(1e-3, rho).

The physical problem and OC update follow Wang et al.; the elasticity and
Helmholtz systems are solved with Topax's structured multigrid solvers.  The
released TOP3D_XL filter element matrix omits the Jacobian scaling from its
gradient term.  On the uniform cubic meshes used here, its nominal filter
radius is therefore represented by a conventional Helmholtz radius larger by
``sqrt(2)``.  This benchmark-local conversion reproduces the released filter
operator without changing Topax's general Helmholtz convention.
"""

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import topax


jax.config.update("jax_enable_x64", True)


# =============================================================================
# Wang et al. Fig. 10 / Table 1 configuration
# =============================================================================

SIZES = np.array([1.0, 0.5, 0.5], dtype=np.float64)

RESOLUTION_ELEMENTS = {
    48: np.array([48, 24, 24], dtype=int),
    96: np.array([96, 48, 48], dtype=int),
    192: np.array([192, 96, 96], dtype=int),
    300: np.array([300, 150, 150], dtype=int),
    384: np.array([384, 192, 192], dtype=int),
    768: np.array([768, 384, 384], dtype=int),
    800: np.array([800, 400, 400], dtype=int),
    1024: np.array([1024, 512, 512], dtype=int),
}

# Deepest exact-dyadic hierarchy that ends at 6 x 3 x 3 for the powers-of-two
# series. The standalone R800 and R1024 defaults use five levels.
DEFAULT_LEVELS = {
    48: 4,
    96: 5,
    192: 6,
    384: 7,
    768: 8,
    800: 5,
    1024: 5,
}

FIXED_FILTER_MULTIPLIER = {
    48: 1,
    96: 2,
    192: 4,
    300: 6,
    384: 8,
    768: 16,
    800: 17,
    1024: 21,
}

# Table 1 values reported for the authors' implementation. These are printed
# as comparison targets; Topax uses a different MGCG implementation.
TABLE_1_STANDARD = {
    48: (965.226, 0.156),
    96: (679.250, 0.091),
    192: (591.716, 0.057),
    300: (566.538, 0.043),
    384: (555.891, 0.036),
    768: (545.192, 0.020),
    800: (545.068, 0.020),
}

TABLE_1_FIXED = {
    48: (965.226, 0.156),
    96: (886.209, 0.143),
    192: (866.852, 0.140),
    300: (843.392, 0.135),
    384: (861.937, 0.139),
    768: (860.676, 0.139),
    800: (861.685, 0.139),
}

ELASTIC_MODULE = 1.0
POISSON_RATIO = 0.3
RHO_MIN = 1.0e-6
SIMP_PENALTY = 3.0

VOLFRAC = 0.12
MAX_OPT_ITER = 50
OC_MOVE = 0.2
OC_BISECTION_TOL = 1.0e-6
OC_PARAMETERS = topax.optimizer.oc.init_oc_params(
    move=OC_MOVE,
    bisect_tol=OC_BISECTION_TOL,
)
OC_STEP = topax.optimizer.oc.construct_oc_step(OC_PARAMETERS)
CHANGE_TOL = 1.0e-4
MDN_TOL = 1.0e-2

CG_TOL = 1.0e-3
FILTER_CG_TOL = CG_TOL / 10.0
CG_MAXITER = 600
JACOBI_DAMPING = 0.5
DEFAULT_N_SMOOTHING_STEPS = 1

PROFILE_ITERATIONS = 5

FIXED_FACE = {
    "center": [0.0, 0.5, 0.5],
    "size": [0.0, 1.0, 1.0],
}

DIRICHLET_BOUNDARY_CONDITIONS = [
    [FIXED_FACE],
    [FIXED_FACE],
    [FIXED_FACE],
]


class BenchmarkConfiguration(NamedTuple):
    resolution: int
    elements: np.ndarray
    n_levels: int
    filter_radius_elements: float
    filter_radius: float
    helmholtz_radius: float
    galerkin_crossover_level: int


def resolve_configuration(
    resolution,
    *,
    n_levels=None,
    fixed_filter=False,
    galerkin_crossover_level=None,
):
    """Validate a paper resolution against Topax's exact dyadic hierarchy."""

    if resolution not in RESOLUTION_ELEMENTS:
        raise ValueError(
            f"resolution must be one of {tuple(RESOLUTION_ELEMENTS)}, got {resolution}."
        )

    if resolution == 300:
        raise ValueError(
            "R300 is intentionally unsupported: 300 x 150 x 150 does not "
            "admit a useful exact-dyadic Topax hierarchy."
        )

    elements = np.array(RESOLUTION_ELEMENTS[resolution], copy=True)

    if n_levels is None:
        n_levels = DEFAULT_LEVELS[resolution]
    if isinstance(n_levels, (bool, np.bool_)) or not isinstance(
        n_levels, (int, np.integer)
    ):
        raise ValueError("n_levels must be a positive integer.")
    n_levels = int(n_levels)
    if n_levels < 2:
        raise ValueError("n_levels must be at least 2 for MGPCG.")

    coarsest_factor = 2 ** (n_levels - 1)
    if np.any(elements % coarsest_factor != 0):
        raise ValueError(
            f"R{resolution} elements {elements.tolist()} are not divisible "
            f"by the exact-dyadic coarsest factor {coarsest_factor}."
        )

    if galerkin_crossover_level is None:
        galerkin_crossover_level = min(4, n_levels - 1)
    if (
        isinstance(galerkin_crossover_level, (bool, np.bool_))
        or not isinstance(galerkin_crossover_level, (int, np.integer))
        or not 1 <= galerkin_crossover_level < n_levels
    ):
        raise ValueError(
            f"galerkin_crossover_level must satisfy 1 <= level < {n_levels}."
        )

    multiplier = FIXED_FILTER_MULTIPLIER[resolution] if fixed_filter else 1
    filter_radius_elements = float(np.sqrt(3.0) * multiplier)
    filter_radius = filter_radius_elements * float(SIZES[0] / elements[0])
    # TOP3D_XL constructs KEF0 without the gradient-term Jacobian factor.
    # For a cubic element, its released matrix is identical up to an overall
    # element-volume factor to Topax's conventional Helmholtz matrix evaluated
    # with sqrt(2) times the nominal physical radius.
    helmholtz_radius = float(np.sqrt(2.0) * filter_radius)

    return BenchmarkConfiguration(
        resolution=resolution,
        elements=elements,
        n_levels=n_levels,
        filter_radius_elements=filter_radius_elements,
        filter_radius=filter_radius,
        helmholtz_radius=helmholtz_radius,
        galerkin_crossover_level=int(galerkin_crossover_level),
    )


# =============================================================================
# Geometry and loading
# =============================================================================


def construct_wang_forces(grid):
    """Construct the normalized -z load over the bottom sixth of x=1."""

    nelx, nely, nelz = map(int, grid.elements)
    n_loaded_z_nodes = int(np.round(nelz / 6.0 + 1.0))
    n_loaded_nodes = (nely + 1) * n_loaded_z_nodes

    forces = jnp.zeros(
        (3, nelx + 1, nely + 1, nelz + 1),
        dtype=jnp.float64,
    )
    forces = forces.at[2, -1, :, :n_loaded_z_nodes].set(-1.0 / n_loaded_nodes)
    return forces, n_loaded_nodes


def construct_problem(configuration):
    elastic = topax.construct_elastic(
        ELASTIC_MODULE,
        POISSON_RATIO,
    )
    grid = topax.construct_grid(
        configuration.n_levels,
        SIZES,
        configuration.elements,
    )
    fine_local_stiffness = topax.construct_local_stiffness(
        grid.fine_level,
        elastic,
    )
    restrictions = topax.construct_restrictions(
        grid,
        elastic,
        DIRICHLET_BOUNDARY_CONDITIONS,
    )
    forces, n_loaded_nodes = construct_wang_forces(grid)

    coarse_physics_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )

    helmholtz = topax.construct_helmholtz(
        filter_radius=configuration.helmholtz_radius,
        thickness=1.0,
    )
    local_helmholtz_operators = tuple(
        topax.construct_local_helmholtz(level, helmholtz) for level in grid.levels
    )
    coarse_helmholtz_structure = topax.sparse.construct_sparse_helmholtz_structure(
        grid.coarse_level,
        local_helmholtz_operators[-1],
    )

    return (
        grid,
        fine_local_stiffness,
        restrictions,
        forces,
        coarse_physics_structure,
        helmholtz,
        local_helmholtz_operators,
        coarse_helmholtz_structure,
        n_loaded_nodes,
    )


# =============================================================================
# Sensitivity filtering and optimality criteria
# =============================================================================


def measure_non_discreteness(rho):
    """Table 1 MDN: zero for binary density and one for rho=0.5."""

    return jnp.mean(4.0 * rho * (1.0 - rho))


def construct_value_and_sensitivity(
    *,
    grid,
    fine_local_stiffness,
    coarse_physics_structure,
    helmholtz,
    local_helmholtz_operators,
    coarse_helmholtz_structure,
    n_smoothing_steps,
    galerkin_crossover_level,
    galerkin_slab_size,
    slab_size,
):
    """Build the benchmark-local elasticity and sensitivity-filter pipeline."""

    fine_level = grid.fine_level

    def value_and_sensitivity(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        physics_value0,
        filter_value0,
    ):
        physics_solver, displacement, physics_info = topax.vcycle.physics_vcycle_solve(
            solver=solvers.physics,
            grid=grid,
            fine_local_operator=fine_local_stiffness,
            coarse_structure=coarse_physics_structure,
            rho=rho,
            forces=forces,
            restrictions=restrictions,
            x0=physics_value0,
            simp_penalty=SIMP_PENALTY,
            rho_min=RHO_MIN,
            damping=JACOBI_DAMPING,
            tol=CG_TOL,
            maxiter=CG_MAXITER,
            n_smoothing_steps=n_smoothing_steps,
            galerkin_crossover_level=galerkin_crossover_level,
            galerkin_slab_size=galerkin_slab_size,
            slab_size=slab_size,
        )
        displacement = jax.lax.stop_gradient(displacement)

        objective = jnp.vdot(forces, displacement).real
        strain_energy = topax.operator.element_strain_energy(
            fine_level,
            displacement,
            fine_local_stiffness,
            restrictions[0],
            slab_size=slab_size,
        )
        positive_stiffness_grad = (
            SIMP_PENALTY
            * (1.0 - RHO_MIN)
            * rho ** (SIMP_PENALTY - 1.0)
            * strain_energy
        )
        raw_objective_grad = -positive_stiffness_grad

        sensitivity_source = rho * raw_objective_grad
        filter_rhs = topax.helmholtz_to_nodal(
            fine_level,
            helmholtz,
            sensitivity_source,
        )
        helmholtz_solver, filtered_nodal, filter_info = (
            topax.vcycle.helmholtz_vcycle_solve(
                solver=solvers.helmholtz,
                grid=grid,
                local_operators=local_helmholtz_operators,
                coarse_structure=coarse_helmholtz_structure,
                rho_mask=rho_mask,
                rho_nodal=filter_rhs,
                x0=filter_value0,
                damping=JACOBI_DAMPING,
                tol=FILTER_CG_TOL,
                maxiter=CG_MAXITER,
                n_smoothing_steps=n_smoothing_steps,
                slab_size=slab_size,
            )
        )
        filtered_sensitivity = topax.helmholtz_to_elemental(
            fine_level,
            filtered_nodal,
        )
        objective_grad = filtered_sensitivity / jnp.maximum(1.0e-3, rho)

        solvers = solvers._replace(
            physics=physics_solver,
            helmholtz=helmholtz_solver,
        )
        return (
            solvers,
            objective,
            objective_grad,
            displacement,
            filtered_nodal,
            physics_info,
            filter_info,
            filter_rhs,
        )

    return value_and_sensitivity


# =============================================================================
# Benchmark driver
# =============================================================================


_RUNTIME_FIELDS = (
    "iteration",
    "compliance",
    "mean_design_density",
    "mean_physical_density",
    "volume_constraint",
    "non_discreteness_reported",
    "change",
    "physics_cg_iterations",
    "physics_relative_residual",
    "filter_cg_iterations",
    "filter_relative_residual",
    "iteration_time_s",
)


def _write_runtime_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_RUNTIME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def optimize(
    *,
    resolution=48,
    n_levels=None,
    fixed_filter=False,
    max_iterations=MAX_OPT_ITER,
    n_smoothing_steps=DEFAULT_N_SMOOTHING_STEPS,
    galerkin_crossover_level=None,
    galerkin_slab_size=topax.galerkin.DEFAULT_GALERKIN_SLAB_SIZE,
    slab_size=topax.operator.DEFAULT_SLAB_SIZE,
    profile=False,
    report_compiler_memory=False,
    runtime_csv=None,
):
    configuration = resolve_configuration(
        resolution,
        n_levels=n_levels,
        fixed_filter=fixed_filter,
        galerkin_crossover_level=galerkin_crossover_level,
    )

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be a positive integer.")
    if not 1 <= max_iterations <= MAX_OPT_ITER:
        raise ValueError(f"max_iterations must satisfy 1 <= value <= {MAX_OPT_ITER}.")
    if (
        isinstance(n_smoothing_steps, (bool, np.bool_))
        or not isinstance(n_smoothing_steps, (int, np.integer))
        or n_smoothing_steps < 1
    ):
        raise ValueError("n_smoothing_steps must be a positive integer.")
    if isinstance(slab_size, bool) or not isinstance(slab_size, int):
        raise ValueError("slab_size must be a positive integer.")
    if slab_size < 1:
        raise ValueError("slab_size must be a positive integer.")
    if (
        isinstance(galerkin_slab_size, bool)
        or not isinstance(galerkin_slab_size, (int, np.integer))
        or galerkin_slab_size < 1
    ):
        raise ValueError("galerkin_slab_size must be a positive integer.")

    setup_start = time.perf_counter()
    (
        grid,
        fine_local_stiffness,
        restrictions,
        forces,
        coarse_physics_structure,
        helmholtz,
        local_helmholtz_operators,
        coarse_helmholtz_structure,
        n_loaded_nodes,
    ) = construct_problem(configuration)

    rho = jnp.full(
        tuple(map(int, configuration.elements)),
        VOLFRAC,
        dtype=jnp.float64,
    )
    rho_mask = jnp.ones_like(rho, dtype=jnp.bool_)
    physics_value0 = jnp.zeros(
        (3, *map(int, grid.fine_level.nodes)),
        dtype=jnp.float64,
    )
    filter_value0 = jnp.zeros(
        (1, *map(int, grid.fine_level.nodes)),
        dtype=jnp.float64,
    )

    solvers = topax.solver.setup_cg_filter_solver_state(
        coarse_physics_structure,
        coarse_helmholtz_structure,
        grid,
        restrictions,
    )

    value_and_sensitivity = construct_value_and_sensitivity(
        grid=grid,
        fine_local_stiffness=fine_local_stiffness,
        coarse_physics_structure=coarse_physics_structure,
        helmholtz=helmholtz,
        local_helmholtz_operators=local_helmholtz_operators,
        coarse_helmholtz_structure=coarse_helmholtz_structure,
        n_smoothing_steps=n_smoothing_steps,
        galerkin_crossover_level=configuration.galerkin_crossover_level,
        galerkin_slab_size=galerkin_slab_size,
        slab_size=slab_size,
    )

    def optimization_step(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        physics_value0,
        filter_value0,
    ):
        (
            solvers,
            objective,
            objective_grad,
            displacement,
            filtered_nodal,
            physics_info,
            filter_info,
            filter_rhs,
        ) = value_and_sensitivity(
            solvers,
            rho,
            forces,
            rho_mask,
            restrictions,
            physics_value0,
            filter_value0,
        )

        volume_constraint = jnp.mean(rho) - VOLFRAC
        rho_new, oc_info = OC_STEP(
            rho,
            objective_grad,
            volume_constraint,
            jnp.ones_like(rho),
        )
        change = oc_info["change"]

        force_norm = jnp.linalg.norm(forces)
        filter_rhs_norm = jnp.linalg.norm(filter_rhs)
        tiny = jnp.finfo(rho.dtype).tiny
        physics_relative_residual = jnp.sqrt(physics_info["error"]) / jnp.maximum(
            force_norm, tiny
        )
        filter_relative_residual = jnp.sqrt(filter_info["error"]) / jnp.maximum(
            filter_rhs_norm, tiny
        )

        diagnostics = (
            objective,
            jnp.mean(rho),
            volume_constraint,
            measure_non_discreteness(rho),
            change,
            physics_info["n_iter"],
            physics_info["converged"],
            physics_relative_residual,
            filter_info["n_iter"],
            filter_info["converged"],
            filter_relative_residual,
        )
        return solvers, rho_new, displacement, filtered_nodal, diagnostics

    optimization_step = jax.jit(
        optimization_step,
        donate_argnums=(1, 5, 6),
    )

    jax.block_until_ready(
        (
            solvers,
            rho,
            forces,
            rho_mask,
            restrictions,
            physics_value0,
            filter_value0,
        )
    )
    setup_time = time.perf_counter() - setup_start

    compilation_start = time.perf_counter()
    compiled_step = optimization_step.lower(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        physics_value0,
        filter_value0,
    ).compile()
    compilation_time = time.perf_counter() - compilation_start

    if report_compiler_memory:
        memory = compiled_step.memory_analysis()
        gib = 1024**3
        print("COMPILER MEMORY ANALYSIS")
        print(f"compiler arguments: {memory.argument_size_in_bytes / gib:10.3f} GiB")
        print(f"compiler output:    {memory.output_size_in_bytes / gib:10.3f} GiB")
        print(f"compiler temporary: {memory.temp_size_in_bytes / gib:10.3f} GiB")
        print(f"compiler aliased:   {memory.alias_size_in_bytes / gib:10.3f} GiB")

    warmup_rho = jnp.array(rho, copy=True)
    warmup_u = jnp.array(physics_value0, copy=True)
    warmup_filter = jnp.array(filter_value0, copy=True)
    warmup_start = time.perf_counter()
    warmup_outputs = compiled_step(
        solvers,
        warmup_rho,
        forces,
        rho_mask,
        restrictions,
        warmup_u,
        warmup_filter,
    )
    jax.block_until_ready(warmup_outputs)
    solvers = warmup_outputs[0]
    warmup_time = time.perf_counter() - warmup_start
    warmup_diagnostics = warmup_outputs[-1]
    if not bool(warmup_diagnostics[6]):
        topax.solver.close_solver_state(solvers)
        raise RuntimeError("Physics CG did not converge during warmup.")
    if not bool(warmup_diagnostics[9]):
        topax.solver.close_solver_state(solvers)
        raise RuntimeError("Sensitivity-filter CG did not converge during warmup.")
    del warmup_outputs, warmup_rho, warmup_u, warmup_filter

    elements = configuration.elements
    n_elements = int(np.prod(elements))
    n_nodes = int(np.prod(elements + 1))
    n_dofs = 3 * n_nodes

    print("WANG ET AL. 2025 FIG. 10 CANTILEVER BENCHMARK")
    print(f"resolution:        R{resolution}")
    print(f"elements:          {tuple(map(int, elements))}")
    print(f"n elements:        {n_elements:,}")
    print(f"elastic DOFs:      {n_dofs:,}")
    print(f"levels:            {configuration.n_levels} (including fine grid)")
    print(f"volume fraction:   {VOLFRAC:.4f}")
    print(
        "TOP3D_XL radius:   "
        f"{configuration.filter_radius_elements:.6f} elements "
        f"({configuration.filter_radius:.8f} physical)"
    )
    print(
        "operator radius:   "
        f"{configuration.helmholtz_radius:.8f} physical "
        "(Topax Helmholtz convention)"
    )
    print(f"filter series:     {'fixed physical' if fixed_filter else 'standard'}")
    print(f"loaded nodes:      {n_loaded_nodes:,}")
    print(f"total force:       {float(jnp.sum(forces)):.6f}")
    print(f"CG tolerance:      {CG_TOL:.1e}")
    print(f"filter tolerance:  {FILTER_CG_TOL:.1e}")
    print(f"Jacobi damping:    {JACOBI_DAMPING:.2f}")
    print(f"MG smoothing:      {n_smoothing_steps} pre + {n_smoothing_steps} post")
    print(
        "Galerkin crossover: "
        f"L{configuration.galerkin_crossover_level} "
        f"(after {configuration.galerkin_crossover_level} coarsenings)"
    )
    print(f"implicit Galerkin: x-slabs of {galerkin_slab_size} elements")
    print(f"matrix-free slab:  {slab_size}")
    print("physics operator:  Topax exact Galerkin MGPCG")
    print("filtering:         PDE sensitivity filter")
    print(f"device:            {jax.devices()[0]}")
    print(f"setup:             {setup_time:10.3f} s")
    print(f"compilation:       {compilation_time:10.3f} s")
    print(f"dummy execution:   {warmup_time:10.3f} s (not timed)")

    profile_dir = f"/tmp/topax-profile-wang-R{resolution}"
    trace_active = False
    if profile:
        shutil.rmtree(profile_dir, ignore_errors=True)
        jax.profiler.start_trace(profile_dir)
        trace_active = True
        print()
        print(f"TensorBoard log:   {profile_dir}")
        print(
            f"profile scope:     iterations 1-{min(PROFILE_ITERATIONS, max_iterations)}"
        )

    iteration_times = []
    physics_iterations = []
    filter_iterations = []
    runtime_rows = []
    converged = False
    last_objective = np.nan
    last_constraint = np.nan
    last_mdn = np.nan
    last_change = np.nan

    try:
        print()
        for iteration in range(1, max_iterations + 1):
            iteration_start = time.perf_counter()
            with jax.profiler.TraceAnnotation(
                "optimization_step",
                iteration=iteration,
            ):
                (
                    solvers,
                    rho,
                    physics_value0,
                    filter_value0,
                    diagnostics,
                ) = compiled_step(
                    solvers,
                    rho,
                    forces,
                    rho_mask,
                    restrictions,
                    physics_value0,
                    filter_value0,
                )
            jax.block_until_ready(
                (solvers, rho, physics_value0, filter_value0, diagnostics)
            )
            iteration_time = time.perf_counter() - iteration_start
            iteration_times.append(iteration_time)

            (
                objective,
                mean_density,
                volume_constraint,
                mdn,
                change,
                physics_n_iter,
                physics_converged,
                physics_relative_residual,
                filter_n_iter,
                filter_converged,
                filter_relative_residual,
            ) = diagnostics

            if not bool(physics_converged):
                raise RuntimeError(
                    f"Physics CG did not converge at iteration {iteration}."
                )
            if not bool(filter_converged):
                raise RuntimeError(
                    f"Sensitivity-filter CG did not converge at iteration {iteration}."
                )

            last_objective = float(objective)
            last_constraint = float(volume_constraint)
            last_mdn = float(mdn)
            last_change = float(change)
            physics_count = int(physics_n_iter)
            filter_count = int(filter_n_iter)
            physics_iterations.append(physics_count)
            filter_iterations.append(filter_count)

            runtime_rows.append(
                {
                    "iteration": iteration,
                    "compliance": last_objective,
                    "mean_design_density": float(mean_density),
                    "mean_physical_density": "",
                    "volume_constraint": last_constraint,
                    "non_discreteness_reported": last_mdn,
                    "change": last_change,
                    "physics_cg_iterations": physics_count,
                    "physics_relative_residual": float(
                        physics_relative_residual
                    ),
                    "filter_cg_iterations": filter_count,
                    "filter_relative_residual": float(filter_relative_residual),
                    "iteration_time_s": iteration_time,
                }
            )

            print(
                f"iter {iteration:3d} | "
                f"c {last_objective:12.6e} | "
                f"rho {float(mean_density):.6f} | "
                f"g {last_constraint:+.3e} | "
                f"MDN {last_mdn:.6f} | "
                f"change {last_change:.3e} | "
                f"CG(K) {physics_count:4d} "
                f"[{float(physics_relative_residual):.2e}] | "
                f"CG(H) {filter_count:4d} "
                f"[{float(filter_relative_residual):.2e}] | "
                f"time {iteration_time:.3f} s"
            )

            if trace_active and iteration == min(
                PROFILE_ITERATIONS,
                max_iterations,
            ):
                jax.profiler.stop_trace()
                trace_active = False

            if last_change <= CHANGE_TOL or last_mdn <= MDN_TOL:
                converged = True
                break
    finally:
        if trace_active:
            jax.profiler.stop_trace()
        topax.solver.close_solver_state(solvers)

    steady_state_time = float(sum(iteration_times))
    average_iteration_time = steady_state_time / len(iteration_times)
    reference_table = TABLE_1_FIXED if fixed_filter else TABLE_1_STANDARD
    reference = reference_table.get(resolution)

    print()
    print("FINAL RESULT")
    print(f"converged:          {converged}")
    print(f"iterations:         {len(iteration_times)}")
    print(f"setup:              {setup_time:10.3f} s")
    print(f"compilation:        {compilation_time:10.3f} s")
    print(f"dummy execution:    {warmup_time:10.3f} s (not timed)")
    print(f"design iterations:  {steady_state_time:10.3f} s")
    print(f"average iteration:  {average_iteration_time:10.3f} s")
    print(f"total CG(K):        {sum(physics_iterations):10d}")
    print(f"average CG(K):      {np.mean(physics_iterations):10.2f}")
    print(f"total CG(H):        {sum(filter_iterations):10d}")
    print(f"average CG(H):      {np.mean(filter_iterations):10.2f}")
    print(f"last compliance:    {last_objective:.10e}")
    print(f"last constraint:    {last_constraint:+.6e}")
    print(f"last MDN:           {last_mdn:.6f}")
    print(f"last change:        {last_change:.6e}")
    print(f"final mean rho:     {float(jnp.mean(rho)):.10f}")
    if reference is not None:
        reference_compliance, reference_mdn = reference
        print(f"Table 1 compliance: {reference_compliance:.6f}")
        print(f"Table 1 MDN:        {reference_mdn:.6f}")

    if runtime_csv is not None:
        _write_runtime_csv(runtime_csv, runtime_rows)

    return {
        "rho": rho,
        "compliance": last_objective,
        "volume_constraint": last_constraint,
        "mdn": last_mdn,
        "iterations": len(iteration_times),
        "configuration": configuration,
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        type=int,
        choices=tuple(RESOLUTION_ELEMENTS),
        default=48,
        help="resolution identifier R; R300 is listed but currently rejected",
    )
    parser.add_argument(
        "--fixed-filter",
        action="store_true",
        help="use the physically fixed filter-radius series from Fig. 10b",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        help=(
            "total exact-dyadic multigrid levels, including the fine grid "
            "(default depends on resolution)"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_OPT_ITER,
        help=f"maximum OC iterations (default: {MAX_OPT_ITER})",
    )
    parser.add_argument(
        "--smoothing-steps",
        type=int,
        default=DEFAULT_N_SMOOTHING_STEPS,
        help="Jacobi sweeps before and after coarse correction",
    )
    parser.add_argument(
        "--galerkin-crossover-level",
        type=int,
        default=None,
        help=(
            "first materialized Galerkin level, with level 0 denoting the "
            "fine grid; defaults to min(4, levels-1)"
        ),
    )
    parser.add_argument(
        "--slab-size",
        type=int,
        default=topax.operator.DEFAULT_SLAB_SIZE,
        help=(
            "matrix-free slab width "
            f"(default: {topax.operator.DEFAULT_SLAB_SIZE})"
        ),
    )
    parser.add_argument(
        "--galerkin-slab-size",
        type=int,
        default=topax.galerkin.DEFAULT_GALERKIN_SLAB_SIZE,
        help=(
            "implicit Galerkin slab width "
            f"(default: {topax.galerkin.DEFAULT_GALERKIN_SLAB_SIZE})"
        ),
    )
    parser.add_argument(
        "--report-compiler-memory",
        action="store_true",
        help="report compiled optimization-step memory statistics",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="capture the first few timed iterations with the JAX profiler",
    )
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        default=None,
        help="write per-design-iteration measurements to this CSV file",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    optimize(
        resolution=args.resolution,
        n_levels=args.levels,
        fixed_filter=args.fixed_filter,
        max_iterations=args.max_iterations,
        n_smoothing_steps=args.smoothing_steps,
        galerkin_crossover_level=args.galerkin_crossover_level,
        galerkin_slab_size=args.galerkin_slab_size,
        slab_size=args.slab_size,
        profile=args.profile,
        report_compiler_memory=args.report_compiler_memory,
        runtime_csv=args.runtime_csv,
    )


if __name__ == "__main__":
    main()
