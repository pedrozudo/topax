"""Träff et al. (2023) 3D cantilever benchmark in Topax.

Reproduces the linear compliance benchmark used in Fig. 6 of:

    E. A. Träff, A. Rydahl, S. Karlsson, O. Sigmund, N. Aage,
    "Simple and efficient GPU accelerated topology optimisation:
    Codes and applications",
    CMAME 410 (2023) 116043.

Benchmark configuration
-----------------------
Fine grid:
    320 x 160 x 160 = 8,192,000 elements

Material:
    E0   = 1.0
    Emin = 1e-6
    nu   = 0.3
    SIMP p = 3

Optimization:
    volume fraction = 0.1
    density filter radius = 1.5 elements
    OC move limit = 0.2
    maximum design iterations = 20

Linear solve:
    PCG tolerance = 1e-5
    damped Jacobi omega = 0.6
    geometric multigrid
    direct Xolky/cuDSS coarse solve

The published Fig. 6 comparison primarily uses four multigrid levels.
The additional best large-grid points use five levels.

Important solver difference
---------------------------
The released Träff OpenMP-GPU code uses four pre- and four post-Jacobi
smoothing sweeps per V-cycle. The Topax solver API defaults to one pre-
and one post-Jacobi sweep for backward compatibility, while this benchmark
defaults to four of each. The sweep count is configurable for direct
comparison.

The fine-grid physics operator uses Topax's slabbed matrix-free
implementation.
"""

import argparse
import csv
import ctypes
import functools
import gc
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import topax


jax.config.update("jax_enable_x64", True)


# =============================================================================
# Träff et al. Fig. 6 benchmark
# =============================================================================

ELEMENTS = np.array([320, 160, 160], dtype=int)

# Their released implementation uses 0.5 x 0.5 x 0.5 elements.
ELEMENT_SIZE = 0.5

SIZES = ELEMENT_SIZE * ELEMENTS.astype(np.float64)

ELASTIC_MODULE = 1.0
POISSON_RATIO = 0.3

SIMP_PENALTY = 3.0
RHO_MIN = 1.0e-6

VOLFRAC = 0.10

FILTER_RADIUS_ELEMENTS = 1.5
FILTER_RADIUS = ELEMENT_SIZE * FILTER_RADIUS_ELEMENTS

CG_TOL = 1.0e-5

# The released top3d.c passes cgmax=200, but the actual GPU CG
# implementation hardcodes maxIter=1000. This reproduces the effective
# solver behavior. Normal convergence should occur far below this anyway.
CG_MAXITER = 1_000

JACOBI_DAMPING = 0.6
DEFAULT_N_SMOOTHING_STEPS = 4
DEFAULT_SLAB_SIZE = topax.operator.DEFAULT_SLAB_SIZE

OC_MOVE = 0.20
OC_BISECTION_TOL = 1.0e-6
OC_PARAMETERS = topax.optimizer.oc.init_oc_params(
    move=OC_MOVE,
    bisect_tol=OC_BISECTION_TOL,
)
OC_STEP = topax.optimizer.oc.construct_oc_step(OC_PARAMETERS)

CHANGE_TOL = 1.0e-2
MAX_OPT_ITER = 20

DEFAULT_N_LEVELS = 4

PROFILE_ITERATIONS = 5


# =============================================================================
# Boundary conditions
# =============================================================================

# The released OpenMP-GPU code does not clamp the entire x=0 face.
#
# It clamps the lower and upper quarter-height strips of the x=0 face,
# spanning the entire z direction.
#
# For nely=320:
#
#     lower strip: y nodes   0 ...  80
#     upper strip: y nodes 240 ... 320
#
# Using normalized Topax boundary coordinates makes the same condition
# propagate naturally to the coarse grids.

_LOWER_SUPPORT = {
    "center": [0.0, 0.125, 0.5],
    "size": [0.0, 0.250, 1.0],
}

_UPPER_SUPPORT = {
    "center": [0.0, 0.875, 0.5],
    "size": [0.0, 0.250, 1.0],
}

DIRICHLET_BOUNDARY_CONDITIONS = [
    [_LOWER_SUPPORT, _UPPER_SUPPORT],
    [_LOWER_SUPPORT, _UPPER_SUPPORT],
    [_LOWER_SUPPORT, _UPPER_SUPPORT],
]


# =============================================================================
# Helpers
# =============================================================================


def non_discreteness(rho):
    """0% for binary density and 100% for rho=0.5."""
    return 100.0 * jnp.mean(4.0 * rho * (1.0 - rho))


def construct_traeff_forces():
    """Construct the circular z-traction patch from the released GPU code.

    The patch lies on the z=0 surface near the free x end.

    The released implementation selects surface elements using

        (i - nelx)^2
        + (j - (nely - 1)/2)^2
        < ((nely + 1)/5)^2

    and distributes a total force of -1 equally over the four nodes of
    every selected surface element.
    """

    nelx, nely, nelz = (
        int(ELEMENTS[0]),
        int(ELEMENTS[1]),
        int(ELEMENTS[2]),
    )

    # Element indices on the z=0 surface.
    element_x = np.arange(
        nelx,
        dtype=np.float64,
    )[:, None]

    element_y = np.arange(
        nely,
        dtype=np.float64,
    )[None, :]

    radius = (nely + 1.0) / 5.0

    center_x = float(nelx)
    center_y = (nely - 1.0) / 2.0

    distance_squared = (element_x - center_x) ** 2 + (element_y - center_y) ** 2

    patch = distance_squared < radius**2

    n_patch_elements = int(np.count_nonzero(patch))

    if n_patch_elements == 0:
        raise RuntimeError("Träff load patch contains no elements.")

    # Every selected surface quad receives one fourth of its share at
    # each corner. Since nodes shared by several selected quads receive
    # contributions from each quad, this reproduces the C implementation.
    nodal_force = -1.0 / (4.0 * n_patch_elements)

    contribution = nodal_force * patch.astype(np.float64)

    face_force = np.zeros(
        (
            nelx + 1,
            nely + 1,
        ),
        dtype=np.float64,
    )

    face_force[:-1, :-1] += contribution
    face_force[1:, :-1] += contribution
    face_force[:-1, 1:] += contribution
    face_force[1:, 1:] += contribution

    # Topax nodal field:
    #
    #     (component, x, y, z)
    #
    # The Träff load acts in the negative z direction on z=0.
    forces = jnp.zeros(
        (
            3,
            nelx + 1,
            nely + 1,
            nelz + 1,
        ),
        dtype=jnp.float64,
    )

    forces = forces.at[
        2,
        :,
        :,
        0,
    ].set(
        jnp.asarray(
            face_force,
            dtype=jnp.float64,
        )
    )

    return forces, n_patch_elements


def construct_problem(n_levels):
    elastic = topax.construct_elastic(
        ELASTIC_MODULE,
        POISSON_RATIO,
    )

    grid = topax.construct_grid(
        n_levels,
        SIZES,
        ELEMENTS,
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

    forces, n_patch_elements = construct_traeff_forces()

    coarse_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )

    conv_filter = topax.construct_conv_filter(
        SIZES,
        ELEMENTS,
        FILTER_RADIUS,
    )

    return (
        grid,
        fine_local_stiffness,
        restrictions,
        forces,
        coarse_structure,
        conv_filter,
        n_patch_elements,
    )


# =============================================================================
# Topax value / sensitivity evaluation
# =============================================================================


def construct_value_and_grad(
    *,
    grid,
    fine_local_stiffness,
    coarse_structure,
    conv_filter,
    n_smoothing_steps,
    galerkin_crossover_level,
    slab_size,
    slab_axis,
):
    """Construct the Träff-style compliance/sensitivity evaluation.

    This is intentionally benchmark-local rather than adding another
    formulation to Topax's dispatcher.

    It returns the physics CG diagnostics explicitly so that the benchmark
    records the number of Krylov iterations.
    """

    fine_level = grid.fine_level
    fine_matvec_impl = functools.partial(
        topax.operator.matrix_free_matvec,
        slab_size=slab_size,
        slab_axis=slab_axis,
    )

    def value_and_grad(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        physics_value0,
    ):
        # ---------------------------------------------------------------------
        # Density filtering
        # ---------------------------------------------------------------------

        def apply_filter(design):
            return conv_filter(
                design,
                rho_mask,
            )

        (
            rho_physical,
            filter_pullback,
        ) = jax.vjp(
            apply_filter,
            rho,
        )

        # ---------------------------------------------------------------------
        # Elasticity solve
        # ---------------------------------------------------------------------

        (
            physics_solver,
            displacement,
            cg_info,
        ) = topax.vcycle.physics_vcycle_solve(
            solver=solvers.physics,
            grid=grid,
            fine_local_operator=fine_local_stiffness,
            coarse_structure=coarse_structure,
            rho=rho_physical,
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
            slab_size=slab_size,
            slab_axis=slab_axis,
        )

        # We explicitly use the displacement as the fixed equilibrium
        # solution when differentiating the elemental stiffness dependence.
        displacement = jax.lax.stop_gradient(displacement)

        # ---------------------------------------------------------------------
        # Compliance and physical-density sensitivity
        # ---------------------------------------------------------------------

        def compliance(
            physical_density,
        ):
            weighing = topax.operator.physics_weighing(
                physical_density,
                SIMP_PENALTY,
                RHO_MIN,
            )

            Ku = fine_matvec_impl(
                fine_level,
                displacement,
                fine_local_stiffness,
                weighing,
                restrictions[0],
            )

            return jnp.vdot(
                displacement,
                Ku,
            ).real

        (
            objective,
            positive_stiffness_grad,
        ) = jax.value_and_grad(compliance)(rho_physical)

        # For compliance:
        #
        #     dC / drho = -u^T (dK/drho) u
        #
        physical_objective_grad = -positive_stiffness_grad

        # ---------------------------------------------------------------------
        # Filter sensitivity
        # ---------------------------------------------------------------------

        # Released code:
        #
        #     dc = H * (dc / Hs)
        #
        # which is precisely the reverse action of the normalized density
        # filter.
        (objective_grad,) = filter_pullback(physical_objective_grad)

        # Released code starts dv=1 and applies the filter-gradient operator:
        #
        #     dv = H * (dv / Hs)
        #
        # This is the derivative of sum(xPhys) with respect to the design.
        (volume_grad,) = filter_pullback(jnp.ones_like(rho_physical))

        solvers = solvers._replace(
            physics=physics_solver,
        )

        return (
            solvers,
            objective,
            objective_grad,
            volume_grad,
            rho_physical,
            displacement,
            cg_info,
        )

    return value_and_grad


# =============================================================================
# Benchmark
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
    n_levels=DEFAULT_N_LEVELS,
    max_iterations=MAX_OPT_ITER,
    n_smoothing_steps=DEFAULT_N_SMOOTHING_STEPS,
    galerkin_crossover_level=None,
    slab_size=DEFAULT_SLAB_SIZE,
    slab_axis=None,
    profile=False,
    report_compiler_memory=False,
    release_initialization_memory=False,
    runtime_csv=None,
):
    if n_levels not in (4, 5):
        raise ValueError("Use 4 or 5 multigrid levels for the Träff Fig. 6 comparison.")

    if galerkin_crossover_level is None:
        galerkin_crossover_level = n_levels - 1
    if (
        isinstance(galerkin_crossover_level, (bool, np.bool_))
        or not isinstance(galerkin_crossover_level, (int, np.integer))
        or not 1 <= galerkin_crossover_level < n_levels
    ):
        raise ValueError(
            f"galerkin_crossover_level must satisfy 1 <= level < {n_levels}."
        )

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    if max_iterations > MAX_OPT_ITER:
        raise ValueError(
            f"max_iterations cannot exceed {MAX_OPT_ITER} for this benchmark."
        )

    if (
        isinstance(n_smoothing_steps, (bool, np.bool_))
        or not isinstance(n_smoothing_steps, (int, np.integer))
        or n_smoothing_steps < 1
    ):
        raise ValueError("n_smoothing_steps must be a positive integer.")

    if (
        isinstance(slab_size, (bool, np.bool_))
        or not isinstance(slab_size, (int, np.integer))
        or slab_size < 1
    ):
        raise ValueError("slab_size must be a positive integer.")
    slab_size = int(slab_size)

    if slab_axis is not None:
        if (
            isinstance(slab_axis, (bool, np.bool_))
            or not isinstance(slab_axis, (int, np.integer))
            or not 0 <= slab_axis < 3
        ):
            raise ValueError("slab_axis must be 0, 1, 2, or None.")
        slab_axis = int(slab_axis)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    setup_start = time.perf_counter()

    (
        grid,
        fine_local_stiffness,
        restrictions,
        forces,
        coarse_structure,
        conv_filter,
        n_patch_elements,
    ) = construct_problem(n_levels)

    rho = jnp.full(
        tuple(int(x) for x in ELEMENTS),
        VOLFRAC,
        dtype=jnp.float64,
    )

    rho_mask = jnp.ones_like(rho)

    physics_value0 = jnp.zeros(
        (
            3,
            *tuple(int(n) for n in grid.fine_level.nodes),
        ),
        dtype=jnp.float64,
    )

    solvers = topax.solver.setup_physics_solver_state(
        coarse_structure, grid, restrictions
    )

    value_and_grad = construct_value_and_grad(
        grid=grid,
        fine_local_stiffness=fine_local_stiffness,
        coarse_structure=coarse_structure,
        conv_filter=conv_filter,
        n_smoothing_steps=n_smoothing_steps,
        galerkin_crossover_level=galerkin_crossover_level,
        slab_size=slab_size,
        slab_axis=slab_axis,
    )

    # -------------------------------------------------------------------------
    # One complete design iteration
    # -------------------------------------------------------------------------

    def optimization_step(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        physics_value0,
    ):
        (
            solvers,
            objective,
            objective_grad,
            volume_grad,
            rho_physical,
            displacement,
            cg_info,
        ) = value_and_grad(
            solvers,
            rho,
            forces,
            rho_mask,
            restrictions,
            physics_value0,
        )

        volume_constraint = jnp.mean(rho_physical) - VOLFRAC

        rho_new, oc_info = OC_STEP(
            rho,
            objective_grad,
            volume_constraint,
            volume_grad,
        )
        change = oc_info["change"]

        force_norm = jnp.sqrt(
            jnp.vdot(
                forces,
                forces,
            ).real
        )

        relative_residual = jnp.sqrt(cg_info["error"]) / force_norm

        diagnostics = (
            objective,
            jnp.mean(rho),
            jnp.mean(rho_physical),
            volume_constraint,
            non_discreteness(rho_physical),
            change,
            cg_info["n_iter"],
            cg_info["converged"],
            relative_residual,
        )

        return (
            solvers,
            rho_new,
            displacement,
            diagnostics,
        )

    # rho and the physics warm start are replaced by their outputs and can
    # therefore be safely donated.
    optimization_step = jax.jit(
        optimization_step,
        donate_argnums=(
            1,
            5,
        ),
    )
    jax.block_until_ready(
        (
            solvers,
            rho,
            forces,
            rho_mask,
            restrictions,
            physics_value0,
        )
    )

    setup_time = time.perf_counter() - setup_start

    # -------------------------------------------------------------------------
    # Explicit compilation
    # -------------------------------------------------------------------------

    compilation_start = time.perf_counter()
    compiled_step = optimization_step.lower(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        physics_value0,
    ).compile()

    compilation_time = time.perf_counter() - compilation_start

    # -------------------------------------------------------------------------
    # Untimed warmup
    #
    # We must preserve Xolky's linear solver state, so retain the solver
    # returned by the warmup. The numerical optimization state is copied and
    # discarded so that timed iteration 1 still starts from rho=volfrac and
    # u0=0 exactly like the reference calculation.
    # -------------------------------------------------------------------------

    warmup_rho = jnp.array(
        rho,
        copy=True,
    )
    warmup_u = jnp.array(
        physics_value0,
        copy=True,
    )
    warmup_start = time.perf_counter()
    warmup_outputs = compiled_step(
        solvers,
        warmup_rho,
        forces,
        rho_mask,
        restrictions,
        warmup_u,
    )

    jax.block_until_ready(warmup_outputs)

    # Keep only the latest native solver resource state.
    solvers = warmup_outputs[0]
    warmup_time = time.perf_counter() - warmup_start

    if not bool(warmup_outputs[-1][-2]):
        topax.solver.close_solver_state(solvers)
        raise RuntimeError("Physics CG did not converge during warmup.")

    del (
        warmup_outputs,
        warmup_rho,
        warmup_u,
    )

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    n_elements = int(np.prod(ELEMENTS))
    n_nodes = int(np.prod(ELEMENTS + 1))
    n_dofs = 3 * n_nodes
    resolved_slab_axis = (
        int(np.argmax(ELEMENTS)) if slab_axis is None else slab_axis
    )
    n_slab_nodes = int(grid.fine_level.nodes[resolved_slab_axis])
    n_slabs = (n_slab_nodes + slab_size - 1) // slab_size

    initialization_gc_objects = ""
    malloc_trim_result = ""
    if release_initialization_memory:
        # The explicitly compiled executable and all numerical state required
        # by the design loop remain live. Drop only Python-side construction
        # and tracing objects that are no longer used after compilation.
        del (
            optimization_step,
            value_and_grad,
            fine_local_stiffness,
            coarse_structure,
            grid,
        )
        initialization_gc_objects = gc.collect()

        # CPython and native libraries may retain otherwise-free heap pages.
        # On glibc, malloc_trim asks the allocator to return such pages to the
        # operating system, making the reduction visible in process RSS.
        try:
            libc = ctypes.CDLL(None)
            malloc_trim = libc.malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            malloc_trim_result = int(malloc_trim(0))
        except (AttributeError, OSError):
            malloc_trim_result = "unavailable"

    print("TRÄFF ET AL. 2023 FIG. 6 CANTILEVER BENCHMARK")
    print(f"elements:          {tuple(int(x) for x in ELEMENTS)}")
    print(f"n elements:        {n_elements:,}")
    print(f"elastic DOFs:      {n_dofs:,}")
    print(f"levels:            {n_levels} (including fine grid)")
    print(f"volume fraction:   {VOLFRAC:.4f}")
    print(
        f"filter radius:     "
        f"{FILTER_RADIUS_ELEMENTS:.2f} elements "
        f"({FILTER_RADIUS:.3f} physical)"
    )
    print(f"load patch elems:  {n_patch_elements:,}")
    print(f"total force:       {float(jnp.sum(forces)):.6f}")
    print(f"CG tolerance:      {CG_TOL:.1e}")
    print(f"Jacobi damping:    {JACOBI_DAMPING:.2f}")
    print(f"MG smoothing:      {n_smoothing_steps} pre + {n_smoothing_steps} post")
    print(
        "Galerkin crossover: "
        f"L{galerkin_crossover_level} "
        f"(after {galerkin_crossover_level} coarsenings)"
    )
    print("physics operator:  matrix_free_matvec")
    print(
        f"fine slabs:        size {slab_size}, "
        f"axis {'xyz'[resolved_slab_axis]} ({resolved_slab_axis}), "
        f"count {n_slabs}"
    )
    print(f"device:            {jax.devices()[0]}")
    print(f"setup:             {setup_time:10.3f} s")
    print(f"compilation:       {compilation_time:10.3f} s")
    print(f"dummy execution:   {warmup_time:10.3f} s (not timed)")
    print(
        "init memory release: "
        f"{'enabled' if release_initialization_memory else 'disabled'}"
    )
    if release_initialization_memory:
        print(f"gc objects:        {initialization_gc_objects}")
        print(f"malloc_trim:       {malloc_trim_result}")

    print()
    if report_compiler_memory:
        memory = compiled_step.memory_analysis()
        gib = 1024**3
        total_size = (
            memory.temp_size_in_bytes
            + memory.argument_size_in_bytes
            + memory.output_size_in_bytes
            - memory.alias_size_in_bytes
        )
        print(f"compiler arguments: {memory.argument_size_in_bytes / gib:10.3f} GiB")
        print(f"compiler output:    {memory.output_size_in_bytes / gib:10.3f} GiB")
        print(f"compiler temporary: {memory.temp_size_in_bytes / gib:10.3f} GiB")
        print(f"compiler aliased:   {memory.alias_size_in_bytes / gib:10.3f} GiB")
        print(f"compiler total:     {total_size / gib:10.3f} GiB")
        print()

    # -------------------------------------------------------------------------
    # Optional TensorBoard profile
    # -------------------------------------------------------------------------

    profile_dir = (
        f"/tmp/topax-profile-traeff-{n_levels}levels-"
        f"slab{slab_size}-axis{resolved_slab_axis}"
    )

    trace_active = False

    if profile:
        shutil.rmtree(
            profile_dir,
            ignore_errors=True,
        )

        jax.profiler.start_trace(profile_dir)

        trace_active = True

        print(f"TensorBoard log:   {profile_dir}")

        print(
            f"profile scope:     iterations 1-{min(PROFILE_ITERATIONS, max_iterations)}"
        )

        print()

    # -------------------------------------------------------------------------
    # Timed design iterations
    # -------------------------------------------------------------------------

    iteration_times = []
    cg_iterations = []
    runtime_rows = []

    converged = False

    last_objective = np.nan
    last_constraint = np.nan
    last_change = np.nan

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        iteration_start = time.perf_counter()

        with jax.profiler.TraceAnnotation(
            "optimization_step",
            iteration=iteration,
        ):
            (
                solvers,
                rho,
                physics_value0,
                diagnostics,
            ) = compiled_step(
                solvers,
                rho,
                forces,
                rho_mask,
                restrictions,
                physics_value0,
            )

        jax.block_until_ready(
            (
                solvers,
                rho,
                physics_value0,
                diagnostics,
            )
        )

        iteration_time = time.perf_counter() - iteration_start
        iteration_times.append(iteration_time)
        (
            objective,
            mean_design_density,
            mean_physical_density,
            volume_constraint,
            non_discreteness_percent,
            change,
            physics_cg_iterations,
            physics_cg_converged,
            relative_residual,
        ) = diagnostics
        objective_value = float(objective)
        constraint_value = float(volume_constraint)
        change_value = float(change)
        cg_count = int(physics_cg_iterations)
        residual_value = float(relative_residual)
        cg_iterations.append(cg_count)
        last_objective = objective_value
        last_constraint = constraint_value
        last_change = change_value

        if not bool(physics_cg_converged):
            if trace_active:
                jax.profiler.stop_trace()
                trace_active = False
            topax.solver.close_solver_state(solvers)
            raise RuntimeError(f"Physics CG did not converge at iteration {iteration}.")

        runtime_rows.append(
            {
                "iteration": iteration,
                "compliance": objective_value,
                "mean_design_density": float(mean_design_density),
                "mean_physical_density": float(mean_physical_density),
                "volume_constraint": constraint_value,
                "non_discreteness_reported": float(non_discreteness_percent),
                "change": change_value,
                "physics_cg_iterations": cg_count,
                "physics_relative_residual": residual_value,
                "filter_cg_iterations": "",
                "filter_relative_residual": "",
                "iteration_time_s": iteration_time,
            }
        )

        print(
            f"iter {iteration:4d} | "
            f"f {objective_value:12.6e} | "
            f"rho {float(mean_design_density):.6f} | "
            f"phys {float(mean_physical_density):.6f} | "
            f"g {constraint_value:+.3e} | "
            f"Mnd "
            f"{float(non_discreteness_percent):8.4f}% | "
            f"change {change_value:.6e} | "
            f"CG(K) {cg_count:4d} | "
            f"relres {residual_value:.3e} | "
            f"time {iteration_time:.3f} s"
        )

        if trace_active and iteration == min(
            PROFILE_ITERATIONS,
            max_iterations,
        ):
            jax.profiler.stop_trace()
            trace_active = False

        if change_value <= CHANGE_TOL:
            converged = True
            break

    if trace_active:
        jax.profiler.stop_trace()

    # -------------------------------------------------------------------------
    # Final physical density
    #
    # This is intentionally outside the benchmark timing and does NOT perform
    # another elasticity solve. The reference implementation likewise updates
    # xPhys after the final OC step without another FE analysis.
    # -------------------------------------------------------------------------

    final_rho_physical = conv_filter(
        rho,
        rho_mask,
    )
    jax.block_until_ready(final_rho_physical)
    final_mean_design = float(jnp.mean(rho))
    final_mean_physical = float(jnp.mean(final_rho_physical))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    steady_state_time = float(sum(iteration_times))
    average_iteration_time = steady_state_time / len(iteration_times)
    throughput = 1.0 / average_iteration_time
    total_cg_iterations = int(sum(cg_iterations))
    average_cg_iterations = float(np.mean(cg_iterations))

    print()
    print("FINAL RESULT")
    print(f"converged:          {converged}")
    print(f"iterations:         {len(iteration_times)}")
    print(f"setup:              {setup_time:10.3f} s")
    print(f"compilation:        {compilation_time:10.3f} s")
    print(f"dummy execution:    {warmup_time:10.3f} s (not timed)")
    print(f"design iterations:  {steady_state_time:10.3f} s")
    print(f"average iteration:  {average_iteration_time:10.3f} s")
    print(f"throughput:         {throughput:10.3f} iterations/s")
    print(f"total CG(K):        {total_cg_iterations:10d}")
    print(f"average CG(K):      {average_cg_iterations:10.2f}")
    print(f"last compliance:    {last_objective:.10e}")
    print(f"last constraint:    {last_constraint:+.6e}")
    print(f"last change:        {last_change:.6e}")
    print(f"final design rho:   {final_mean_design:.10f}")
    print(f"final physical rho: {final_mean_physical:.10f}")

    topax.solver.close_solver_state(solvers)

    if runtime_csv is not None:
        _write_runtime_csv(runtime_csv, runtime_rows)

    return (
        rho,
        final_rho_physical,
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--levels",
        type=int,
        choices=(4, 5),
        default=DEFAULT_N_LEVELS,
        help=(
            "total multigrid levels including the fine grid; use 4 for the "
            "main Fig. 6 comparison or 5 for the largest grids "
            f"(default: {DEFAULT_N_LEVELS})"
        ),
    )

    parser.add_argument(
        "--elements",
        type=int,
        nargs=3,
        default=tuple(map(int, ELEMENTS)),
        metavar=("EX", "EY", "EZ"),
        help="fine-grid element counts (default: 320 160 160)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_OPT_ITER,
        help=(f"maximum OC design iterations (default: {MAX_OPT_ITER})"),
    )

    parser.add_argument(
        "--smoothing-steps",
        type=int,
        default=DEFAULT_N_SMOOTHING_STEPS,
        help=(
            "Jacobi sweeps before and after coarse correction "
            f"(default: {DEFAULT_N_SMOOTHING_STEPS})"
        ),
    )

    parser.add_argument(
        "--galerkin-crossover-level",
        type=int,
        default=None,
        help=(
            "first materialized level, with level 0 denoting the fine grid; "
            "defaults to the coarsest level"
        ),
    )

    parser.add_argument(
        "--slab-size",
        type=int,
        default=DEFAULT_SLAB_SIZE,
        help=f"fine-grid output-node slab size (default: {DEFAULT_SLAB_SIZE})",
    )

    parser.add_argument(
        "--slab-axis",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="fine-grid slab axis: 0=x, 1=y, 2=z; default selects largest axis",
    )

    parser.add_argument(
        "--report-compiler-memory",
        action="store_true",
        help="report compiled optimization-step memory statistics",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help=("capture the first few timed iterations with the JAX profiler"),
    )

    parser.add_argument(
        "--release-initialization-memory",
        action="store_true",
        help=(
            "drop unused Python setup/tracing objects and request a glibc "
            "heap trim after compilation and warmup"
        ),
    )

    parser.add_argument(
        "--runtime-csv",
        type=Path,
        default=None,
        help="write per-design-iteration measurements to this CSV file",
    )

    return parser.parse_args(argv)


def main():
    global ELEMENTS, SIZES

    args = parse_args()

    ELEMENTS = np.asarray(args.elements, dtype=int)
    if np.any(ELEMENTS <= 0):
        raise ValueError("elements must be positive integers.")
    SIZES = ELEMENT_SIZE * ELEMENTS.astype(np.float64)

    optimize(
        n_levels=args.levels,
        max_iterations=args.max_iterations,
        n_smoothing_steps=args.smoothing_steps,
        galerkin_crossover_level=args.galerkin_crossover_level,
        slab_size=args.slab_size,
        slab_axis=args.slab_axis,
        profile=args.profile,
        report_compiler_memory=args.report_compiler_memory,
        release_initialization_memory=args.release_initialization_memory,
        runtime_csv=args.runtime_csv,
    )


if __name__ == "__main__":
    main()
