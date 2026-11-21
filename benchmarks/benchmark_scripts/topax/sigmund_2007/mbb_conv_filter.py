import argparse
import csv
import json
import os
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

import topax

from topax import (
    FILTER_CONVOLUTION,
    FORMULATION_MMA,
    PROBLEM_COMPLIANCE,
)


jax.config.update(
    "jax_enable_x64",
    True,
)


# =============================================================================
# Sigmund (2007)
#
# "Morphology-based black and white filters for topology optimization"
#
# MBB beam — standard density filter with linearly decaying weighting
#
# Literature setup:
#
#     mesh:               120 x 40
#     E0:                 1
#     Emin:               1e-9
#     nu:                 0.3
#     SIMP p:             3
#     volume fraction:    0.5
#     filter radius:      3.5
#
# Density filter:
#
#     rho
#      |
#      v
#   density filter
#   linear weighting
#   R = 3.5
#      |
#      v
#   rho_physical
#      |
#      v
#     SIMP
#      |
#      v
#   elasticity
#
# Literature result:
#
#     compliance:         218.3
#     Mnd:                24.02 %
#     iterations:         1000
#
# =============================================================================


# =============================================================================
# Mesh cases
# =============================================================================


@dataclass(frozen=True)
class MeshCase:
    nx: int
    ny: int
    n_levels: int
    reference_compliance: float | None = None
    reference_mnd: float | None = None
    reference_iterations: int | None = None

    @property
    def label(self):
        return f"{self.nx}x{self.ny}"

    @property
    def elements(self):
        return np.array([self.nx, self.ny], dtype=np.int64)


CASES = (
    MeshCase(120, 40, 2, 218.3, 24.02, 1000),
    MeshCase(360, 120, 2),
    MeshCase(750, 250, 2),
    MeshCase(1500, 500, 3),
    MeshCase(3000, 1000, 3),
    MeshCase(5400, 1800, 3),
)

CASE_BY_LABEL = {case.label: case for case in CASES}
DEFAULT_CASE = CASES[0]


# =============================================================================
# Physical problem
# =============================================================================

# The scaled exposition cases retain unit-sized elements, as in the paper
# case, so the density-filter radius remains 3.5 elements.
SIZES = DEFAULT_CASE.elements.astype(np.float64)

ELEMENTS = DEFAULT_CASE.elements

ELASTIC_MODULE = 1.0
POISSON_RATIO = 0.3
THICKNESS = 1.0

SIMP_PENALTY = 3
RHO_MIN = 1e-9

VOLFRAC = 0.5


# =============================================================================
# Density filter
#
# With SIZES = [120, 40] and ELEMENTS = [120, 40], each element is 1 x 1.
# Therefore Sigmund's R = 3.5 can be used directly.
# =============================================================================

FILTER_RADIUS = 3.5


# =============================================================================
# Linear solver
#
# These are TopaX numerical choices, not parameters from the paper.
#
#     120 x 40
#      60 x 20
# =============================================================================

N_LEVELS = DEFAULT_CASE.n_levels

TOL = 1e-10
MAXITER = 500

DAMPING = 0.8


# =============================================================================
# MMA
# =============================================================================

MMA_MOVE = 0.5

CHANGE_TOL = 0.01
MAX_OPT_ITER = 10_000

PLATEAU_MIN_ITER = 500
PLATEAU_WINDOW = 100
PLATEAU_REL_TOL = 1.0e-5


# =============================================================================
# Literature reference
# =============================================================================

REFERENCE_COMPLIANCE = DEFAULT_CASE.reference_compliance

REFERENCE_MEAN_RHO = 0.5

REFERENCE_MND = DEFAULT_CASE.reference_mnd
REFERENCE_ITERATIONS = DEFAULT_CASE.reference_iterations


# =============================================================================
# Console output
# =============================================================================

PRINT_EVERY = 10


# =============================================================================
# Output
# =============================================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "benchmarks/results/sigmund_2007"
)
DESIGN_DENSITY_FILE = OUTPUT_DIR / "mbb_density_filter_rho.npy"
PHYSICAL_DENSITY_FILE = OUTPUT_DIR / "mbb_density_filter_rho_physical.npy"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"


def configure_output_dir(output_dir):
    global OUTPUT_DIR
    global DESIGN_DENSITY_FILE
    global PHYSICAL_DENSITY_FILE
    global SUMMARY_CSV
    global SUMMARY_JSON

    OUTPUT_DIR = Path(output_dir).expanduser().resolve()
    DESIGN_DENSITY_FILE = OUTPUT_DIR / "mbb_density_filter_rho.npy"
    PHYSICAL_DENSITY_FILE = OUTPUT_DIR / "mbb_density_filter_rho_physical.npy"
    SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
    SUMMARY_JSON = OUTPUT_DIR / "summary.json"


def configure_case(case, n_levels=None):
    global SIZES
    global ELEMENTS
    global N_LEVELS
    global REFERENCE_COMPLIANCE
    global REFERENCE_MND
    global REFERENCE_ITERATIONS

    SIZES = case.elements.astype(np.float64)
    ELEMENTS = case.elements
    N_LEVELS = case.n_levels if n_levels is None else n_levels
    REFERENCE_COMPLIANCE = case.reference_compliance
    REFERENCE_MND = case.reference_mnd
    REFERENCE_ITERATIONS = case.reference_iterations


# =============================================================================
# Helpers
# =============================================================================


def identity_projection(x):
    """
    No Heaviside projection is used for this benchmark.

    Sigmund Table 1:
        "Density w. lin. weight"

    corresponds only to the standard density filter.
    """

    return x


def non_discreteness(rho_physical):
    """
    Sigmund's non-discreteness measure:

        Mnd = 100 / n * sum_e 4 rho_tilde_e (1 - rho_tilde_e)

    For density filtering, this must be evaluated using the filtered
    physical density, not the unfiltered design variables.
    """

    return 100.0 * jnp.mean(4.0 * rho_physical * (1.0 - rho_physical))


def print_force_diagnostics(
    forces,
    restrictions,
):
    """
    Verify that the MBB point load was actually created and is not
    eliminated by the Dirichlet restrictions.
    """

    forces.block_until_ready()

    force_norm = jnp.linalg.norm(forces)
    force_abs_sum = jnp.sum(jnp.abs(forces))
    force_nonzero = jnp.count_nonzero(forces)
    force_indices = jnp.argwhere(forces != 0)
    fine_restriction = restrictions[0]
    active_forces = forces * fine_restriction
    active_force_norm = jnp.linalg.norm(active_forces)

    jax.block_until_ready(
        (
            force_norm,
            force_abs_sum,
            force_nonzero,
            force_indices,
            active_force_norm,
        )
    )

    print()
    print("=" * 80)
    print("FORCE CHECK")
    print("=" * 80)

    print(f"shape:                 {forces.shape}")
    print(f"min:                   {float(jnp.min(forces)):.16e}")
    print(f"max:                   {float(jnp.max(forces)):.16e}")
    print(f"norm:                  {float(force_norm):.16e}")
    print(f"sum(abs):              {float(force_abs_sum):.16e}")
    print(f"nonzero entries:       {int(force_nonzero)}")
    print("nonzero indices:")
    print(np.asarray(force_indices))
    if int(force_nonzero) > 0:
        print("nonzero values:")
        print(np.asarray(forces[tuple(force_indices.T)]))
    print()
    print(f"norm after restriction: {float(active_force_norm):.16e}")

    # -------------------------------------------------------------------------
    # Fail immediately for an invalid benchmark setup.
    # -------------------------------------------------------------------------

    if int(force_nonzero) == 0:
        raise RuntimeError(
            "MBB force construction produced an entirely zero force vector."
        )

    if not np.isclose(
        float(force_norm),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"MBB force vector does not have unit norm. Got {float(force_norm):.16e}."
        )

    if not np.isclose(
        float(active_force_norm),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "MBB force is partially or fully removed by the "
            "Dirichlet restrictions. "
            f"Norm after restriction is "
            f"{float(active_force_norm):.16e}."
        )

    print()
    print("force check:          PASS")


def print_restriction_diagnostics(
    restrictions,
):
    print()
    print("=" * 80)
    print("RESTRICTION CHECK")
    print("=" * 80)

    for level_index, restriction in enumerate(restrictions):
        restriction.block_until_ready()

        free = int(jnp.count_nonzero(restriction))

        total = int(restriction.size)

        restricted = total - free

        print(
            f"level {level_index}: "
            f"shape={restriction.shape}, "
            f"free={free}, "
            f"restricted={restricted}"
        )

    # -------------------------------------------------------------------------
    # Fine-grid expectation:
    #
    # ux = 0 for all 41 nodes on x = 0
    # uy = 0 for one node at the lower-right corner
    #
    # Therefore we expect 42 restricted displacement DOFs.
    # -------------------------------------------------------------------------

    fine = restrictions[0]

    n_restricted = fine.size - int(jnp.count_nonzero(fine))

    print()

    print(f"fine restricted DOFs:  {n_restricted}")

    print("expected:              42")

    if n_restricted != 42:
        print(
            "WARNING: fine-grid restricted DOF count "
            "does not match the expected MBB value."
        )

    else:
        print("restriction count:     PASS")


def print_filter_diagnostics(
    conv_filter,
    rho,
    rho_mask,
):
    rho_physical = conv_filter(
        rho,
        rho_mask,
    )
    rho_physical.block_until_ready()
    rho_min = float(jnp.min(rho_physical))
    rho_max = float(jnp.max(rho_physical))
    rho_mean = float(jnp.mean(rho_physical))
    max_abs_difference = float(jnp.max(jnp.abs(rho_physical - rho)))

    print()
    print("=" * 80)
    print("FILTER CHECK")
    print("=" * 80)
    print(f"filter radius:         {FILTER_RADIUS}")
    print(
        f"design min/max:        "
        f"{float(jnp.min(rho)):.16e} / "
        f"{float(jnp.max(rho)):.16e}"
    )
    print(f"physical min/max:      {rho_min:.16e} / {rho_max:.16e}")
    print(f"physical mean:         {rho_mean:.16e}")
    print(f"max |physical-design|: {max_abs_difference:.16e}")

    if not np.allclose(
        np.asarray(rho_physical),
        VOLFRAC,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("Uniform density was not preserved by the density filter.")

    print()
    print("filter check:         PASS")


def print_initial_value_and_grad_diagnostics(
    objective,
    constraint,
):
    objective_grad_norm = jnp.linalg.norm(objective.grad)

    constraint_grad_norm = jnp.linalg.norm(constraint.grad)

    jax.block_until_ready(
        (
            objective.value,
            objective.grad,
            constraint.value,
            constraint.grad,
            objective_grad_norm,
            constraint_grad_norm,
        )
    )

    print()
    print("=" * 80)
    print("INITIAL VALUE + GRAD CHECK")
    print("=" * 80)

    print(f"objective:             {float(objective.value):.16e}")
    print(f"objective grad min:    {float(jnp.min(objective.grad)):.16e}")
    print(f"objective grad max:    {float(jnp.max(objective.grad)):.16e}")
    print(f"objective grad norm:   {float(objective_grad_norm):.16e}")
    print()
    print(f"constraint:            {float(constraint.value):.16e}")
    print(f"constraint grad min:   {float(jnp.min(constraint.grad)):.16e}")
    print(f"constraint grad max:   {float(jnp.max(constraint.grad)):.16e}")
    print(f"constraint grad norm:  {float(constraint_grad_norm):.16e}")

    # -------------------------------------------------------------------------
    # A valid loaded elasticity problem cannot have zero compliance.
    # -------------------------------------------------------------------------

    objective_value = float(objective.value)

    objective_gradient_norm = float(objective_grad_norm)

    if not np.isfinite(objective_value):
        raise RuntimeError("Initial compliance is not finite.")

    if objective_value <= 0.0:
        raise RuntimeError(
            "Initial MBB compliance is zero or negative. "
            "The physics problem is not configured correctly."
        )

    if not np.isfinite(objective_gradient_norm):
        raise RuntimeError("Initial compliance gradient is not finite.")

    if objective_gradient_norm == 0.0:
        raise RuntimeError("Initial compliance gradient is identically zero.")

    print()
    print("initial value+grad:    PASS")


def save_densities(
    rho,
    rho_physical,
):
    """
    Save both fields.

    rho:
        Mathematical MMA design variables.

    rho_physical:
        Filtered density used by SIMP and therefore the field that should
        be compared physically with Sigmund's published result.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rho_np = np.asarray(rho)

    rho_physical_np = np.asarray(rho_physical)

    np.save(
        DESIGN_DENSITY_FILE,
        rho_np,
    )

    np.save(
        PHYSICAL_DENSITY_FILE,
        rho_physical_np,
    )

    print()

    print(f"saved design density:   {DESIGN_DENSITY_FILE}")

    print(f"saved physical density: {PHYSICAL_DENSITY_FILE}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sigmund (2007) MBB validation benchmark."
    )
    parser.add_argument(
        "--case",
        choices=tuple(case.label for case in CASES),
        default=DEFAULT_CASE.label,
        help="Mesh to run (default: 120x40 literature case)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX platform: CPU or NVIDIA CUDA GPU",
    )
    parser.add_argument(
        "--n-levels",
        type=int,
        choices=(2, 3, 4),
        help="Override the case's default multigrid hierarchy depth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for densities and summaries",
    )
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        help="write per-design-iteration measurements to this CSV file",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_OPT_ITER,
        help=f"maximum optimization iterations (default: {MAX_OPT_ITER})",
    )
    return parser.parse_args(argv)


def runtime_metadata():
    device = jax.devices()[0]
    memory_stats = {}
    try:
        memory_stats = device.memory_stats() or {}
    except (AttributeError, RuntimeError):
        pass

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        peak_rss /= 1024.0

    return {
        "device_platform": device.platform,
        "device_kind": getattr(device, "device_kind", str(device)),
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "xla_python_client_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE", ""
        ),
        "host_peak_rss_kb_internal": int(peak_rss),
        "allocator_peak_bytes_in_use": memory_stats.get(
            "peak_bytes_in_use", ""
        ),
        "allocator_bytes_limit": memory_stats.get("bytes_limit", ""),
    }


def save_summary(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    payload = {
        "paper": "Sigmund (2007), density-filter MBB benchmark",
        "cases": [result],
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"saved summary CSV:      {SUMMARY_CSV}")
    print(f"saved summary JSON:     {SUMMARY_JSON}")


def save_history(history, path=None):
    path = OUTPUT_DIR / "history.csv" if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"saved iteration history: {path}")


# =============================================================================
# Problem construction
# =============================================================================


def construct_problem():
    # =========================================================================
    # Elasticity
    # =========================================================================

    elastic = topax.construct_elastic(
        ELASTIC_MODULE,
        POISSON_RATIO,
        THICKNESS,
    )

    # =========================================================================
    # Grid
    # =========================================================================

    grid = topax.construct_grid(
        N_LEVELS,
        SIZES,
        ELEMENTS,
    )

    # =========================================================================
    # Local stiffness operators
    # =========================================================================

    fine_local_stiffness = topax.construct_local_stiffness(
        grid.fine_level,
        elastic,
    )

    # =========================================================================
    # MBB boundary conditions
    #
    # Region selectors are normalized coordinates.
    #
    # Half-domain:
    #
    #     left edge:
    #         ux = 0
    #
    #     lower-right corner:
    #         uy = 0
    # =========================================================================

    dirichlet_boundary_conditions = [
        # ux = 0 on left edge
        [
            {
                "center": [
                    0.0,
                    0.5,
                ],
                "size": [
                    0.0,
                    1.0,
                ],
            },
        ],
        # uy = 0 at lower-right corner
        [
            {
                "center": [
                    1.0,
                    0.0,
                ],
                "size": [
                    0.0,
                    0.0,
                ],
            },
        ],
    ]

    restrictions = topax.construct_restrictions(
        grid,
        elastic,
        dirichlet_boundary_conditions,
    )

    # =========================================================================
    # MBB load
    #
    # Downward unit point load at the upper-left node.
    #
    # Region selectors are normalized coordinates.
    # =========================================================================

    concentrated_forces = [
        [],
        [
            {
                "center": [
                    0.0,
                    1.0,
                ],
                "size": [
                    0.0,
                    0.0,
                ],
                "intensity": -1.0,
            },
        ],
    ]

    forces = topax.construct_forces(
        grid,
        elastic,
        concentrated_forces,
    )

    # =========================================================================
    # Density filter
    #
    # Sigmund Eq. (12), with the linearly decaying weighting of Eq. (13).
    #
    # Because the physical element dimensions are 1 x 1 here, the paper's
    # R = 3.5 is used directly.
    # =========================================================================

    conv_filter = topax.construct_conv_filter(
        SIZES,
        ELEMENTS,
        FILTER_RADIUS,
    )

    # =========================================================================
    # Validate BCs and load BEFORE compiling the optimization.
    # =========================================================================

    print_restriction_diagnostics(restrictions)

    print_force_diagnostics(
        forces,
        restrictions,
    )

    # =========================================================================
    # Sparse coarse physics structure
    # =========================================================================

    start = time.perf_counter()

    coarse_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )

    jax.effects_barrier()

    elapsed = time.perf_counter() - start

    print()

    print(f"coarse structure construction: {1000.0 * elapsed:.4f} ms")

    return (
        grid,
        fine_local_stiffness,
        restrictions,
        forces,
        coarse_structure,
        conv_filter,
    )


# =============================================================================
# Main
# =============================================================================


def main(argv=None):
    global MAX_OPT_ITER

    args = parse_args(argv)
    if not 1 <= args.max_iterations <= MAX_OPT_ITER:
        raise ValueError(
            f"max_iterations must satisfy 1 <= value <= {MAX_OPT_ITER}."
        )
    MAX_OPT_ITER = args.max_iterations
    case = CASE_BY_LABEL[args.case]
    configure_case(case, args.n_levels)
    platform_name = "cuda" if args.device == "gpu" else "cpu"
    jax.config.update("jax_platform_name", platform_name)
    configure_output_dir(args.output_dir)

    case_wall_start = time.perf_counter()

    print()
    print("=" * 80)
    print("SIGMUND (2007) MBB — DENSITY FILTER WITH LINEAR WEIGHTING")
    print("=" * 80)

    print(f"case:              {case.label}")
    print(f"elements:          {tuple(ELEMENTS)}")
    print(f"sizes:             {tuple(SIZES)}")
    print(f"E0:                {ELASTIC_MODULE}")
    print(f"Emin:              {RHO_MIN}")
    print(f"Poisson ratio:     {POISSON_RATIO}")
    print(f"SIMP penalty:      {SIMP_PENALTY}")
    print(f"volume fraction:   {VOLFRAC}")
    print(f"filter radius:     {FILTER_RADIUS}")
    print(f"MMA move:          {MMA_MOVE}")
    print(f"change tolerance:  {CHANGE_TOL}")
    print(
        "compliance plateau: "
        f"{PLATEAU_REL_TOL:.1e} over {PLATEAU_WINDOW} iterations "
        f"after iteration {PLATEAU_MIN_ITER}"
    )
    print(f"maximum iterations: {MAX_OPT_ITER}")
    print(f"MG levels:         {N_LEVELS}")
    print(f"CG tolerance:      {TOL}")
    print(f"device:            {jax.devices()[0]}")
    print(f"x64 enabled:       {jax.config.jax_enable_x64}")

    # =========================================================================
    # Construct physical problem
    # =========================================================================

    setup_start = time.perf_counter()

    (
        grid,
        fine_local_stiffness,
        restrictions,
        forces,
        coarse_structure,
        conv_filter,
    ) = construct_problem()

    solvers = topax.solver.setup_physics_solver_state(
        coarse_structure,
        grid,
        restrictions,
    )

    # =========================================================================
    # Initial design variables
    # =========================================================================

    rho = VOLFRAC * jnp.ones(
        ELEMENTS,
        dtype=jnp.float64,
    )

    # No fixed-solid / fixed-void design region.
    #
    # Keep this in the standard dynamic TopaX interface.

    rho_mask = jnp.ones_like(
        rho,
        dtype=jnp.bool_,
    )

    # =========================================================================
    # Validate density filter
    # =========================================================================

    print_filter_diagnostics(conv_filter, rho, rho_mask)

    # =========================================================================
    # TopaX value + gradient
    #
    # FILTER_CONVOLUTION:
    #
    #       rho
    #        |
    #        v
    #   density filter
    #      R = 3.5
    #        |
    #        v
    #   rho_physical
    #        |
    #        v
    #       SIMP
    #        |
    #        v
    #     elasticity
    #
    # No Heaviside projection is used.
    #
    # Both compliance and the volume constraint operate on rho_physical.
    # =========================================================================

    (
        value_and_grad,
        initial_cg_state,
    ) = topax.construct_value_and_grad(
        problem=PROBLEM_COMPLIANCE,
        filtering=FILTER_CONVOLUTION,
        formulation=FORMULATION_MMA,
        constraint_function=topax.normalized_volume_constraint,
        fine_local_operator=fine_local_stiffness,
        conv_filter=conv_filter,
        projection=identity_projection,
        grid_physics=grid,
        coarse_physics_structure=coarse_structure,
        simp_penalty=SIMP_PENALTY,
        rho_min=RHO_MIN,
        tol=TOL,
        maxiter=MAXITER,
        damping=DAMPING,
        volfrac=VOLFRAC,
    )

    value_and_grad = jax.jit(value_and_grad)

    # =========================================================================
    # MMA
    # =========================================================================

    mma_parameters = topax.optimizer.mma.init_mma_1con_params(
        move=MMA_MOVE,
    )

    mma_state = topax.optimizer.mma.init_mma_1con_state(
        rho,
        mma_parameters,
    )

    mma_step_impl = topax.optimizer.mma.construct_mma_step(
        mma_parameters=mma_parameters,
    )

    def mma_step(
        rho,
        objective_value,
        objective_grad,
        constraint_value,
        constraint_grad,
        mma_state,
    ):
        (
            rho_new,
            mma_state_new,
            info,
        ) = mma_step_impl(
            rho,
            objective_value,
            objective_grad,
            constraint_value,
            constraint_grad,
            mma_state,
        )

        change = jnp.max(jnp.abs(rho_new - rho))

        return (
            rho_new,
            mma_state_new,
            info,
            change,
        )

    mma_step = jax.jit(
        mma_step,
        donate_argnums=(0,),
    )

    setup_time = time.perf_counter() - setup_start

    # =========================================================================
    # Explicit compilation and untimed warm-up
    # =========================================================================

    print("Compiling initial evaluation...", flush=True)
    compilation_start = time.perf_counter()
    compiled_evaluation = value_and_grad.lower(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        initial_cg_state,
    ).compile()
    compilation_time = time.perf_counter() - compilation_start
    print(f"Initial evaluation compiled in {compilation_time:.3f} s")

    print("Running untimed initial evaluation...", flush=True)
    initial_evaluation_start = time.perf_counter()

    (
        solvers,
        initial_objective,
        initial_constraint,
        evaluated_cg_state,
        initial_linear_solve_diagnostics,
    ) = compiled_evaluation(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        initial_cg_state,
    )

    jax.block_until_ready(
        (
            initial_objective.value,
            initial_objective.grad,
            initial_constraint.value,
            initial_constraint.grad,
            evaluated_cg_state,
            initial_linear_solve_diagnostics.converged,
        )
    )
    if not bool(initial_linear_solve_diagnostics.converged):
        raise RuntimeError("Linear solve did not converge during warm-up.")

    initial_evaluation_time = time.perf_counter() - initial_evaluation_start

    print_initial_value_and_grad_diagnostics(
        initial_objective,
        initial_constraint,
    )
    objective_scale = float(initial_objective.value)

    # Match the Träff/Wang protocol: retain the native solver resources from
    # the untimed evaluation, but start the timed numerical state from the
    # original warm start.
    cg_state = initial_cg_state
    del evaluated_cg_state

    def optimization_step(
        solvers,
        rho,
        cg_state,
        mma_state,
    ):
        (
            solvers,
            objective,
            constraint,
            cg_state,
            linear_solve_diagnostics,
        ) = value_and_grad(
            solvers,
            rho,
            forces,
            rho_mask,
            restrictions,
            cg_state,
        )
        rho_new, mma_state, mma_info, change = mma_step(
            rho,
            objective.value,
            objective.grad,
            constraint.value,
            constraint.grad,
            mma_state,
        )
        rho_physical = conv_filter(rho_new, rho_mask)
        diagnostics = (
            objective.value,
            constraint.value,
            change,
            jnp.mean(rho_new),
            jnp.mean(rho_physical),
            jnp.mean(rho_physical) / VOLFRAC,
            non_discreteness(rho_physical),
            mma_info["feasible"],
            mma_info["bracketed"],
            mma_info["lam"],
            mma_info["phi"],
            linear_solve_diagnostics.physics["n_iter"],
            linear_solve_diagnostics.converged,
            linear_solve_diagnostics.physics["relative_residual"],
        )
        return solvers, rho_new, cg_state, mma_state, diagnostics

    optimization_step = jax.jit(optimization_step)

    print("Compiling complete optimization step...", flush=True)
    compilation_start = time.perf_counter()
    compiled_step = optimization_step.lower(
        solvers,
        rho,
        cg_state,
        mma_state,
    ).compile()
    step_compilation_time = time.perf_counter() - compilation_start
    compilation_time += step_compilation_time
    print(f"Optimization step compiled in {step_compilation_time:.3f} s")

    warmup_rho = jnp.array(rho, copy=True)
    warmup_cg_state = jax.tree.map(
        lambda value: jnp.array(value, copy=True),
        cg_state,
    )
    warmup_mma_state = jax.tree.map(
        lambda value: jnp.array(value, copy=True),
        mma_state,
    )
    print("Running untimed dummy iteration...", flush=True)
    warmup_start = time.perf_counter()
    warmup_outputs = compiled_step(
        solvers,
        warmup_rho,
        warmup_cg_state,
        warmup_mma_state,
    )
    jax.block_until_ready(warmup_outputs)
    solvers = warmup_outputs[0]
    warmup_diagnostics = warmup_outputs[-1]
    warmup_time = time.perf_counter() - warmup_start
    print(f"Dummy iteration completed in {warmup_time:.3f} s")
    if not bool(warmup_diagnostics[12]):
        raise RuntimeError("Linear solve did not converge during dummy execution.")
    if not bool(warmup_diagnostics[7]):
        raise RuntimeError("MMA failed during dummy execution.")
    del (
        warmup_outputs,
        warmup_rho,
        warmup_cg_state,
        warmup_mma_state,
    )

    compile_warmup_time = (
        compilation_time + initial_evaluation_time + warmup_time
    )

    # =========================================================================
    # Optimization state
    # =========================================================================

    converged = False
    final_change = np.inf
    final_plateau_relative_change = None
    stopping_reason = "iteration_limit"
    history = []
    iteration_times = []

    # =========================================================================
    # Optimization
    # =========================================================================

    print()
    print("=" * 80)
    print("OPTIMIZATION")
    print("=" * 80)

    loop_start = time.perf_counter()

    for iteration in range(
        1,
        MAX_OPT_ITER + 1,
    ):
        iteration_start = time.perf_counter()
        solvers, rho, cg_state, mma_state, diagnostics = compiled_step(
            solvers,
            rho,
            cg_state,
            mma_state,
        )
        jax.block_until_ready(
            (solvers, rho, cg_state, mma_state, diagnostics)
        )
        iteration_time = time.perf_counter() - iteration_start
        iteration_times.append(iteration_time)

        (
            objective_value,
            constraint_value,
            change,
            mean_design_rho,
            mean_physical_rho,
            volume_ratio,
            mnd,
            mma_feasible,
            mma_bracketed,
            mma_lambda,
            mma_phi,
            physics_cg_iterations,
            physics_cg_converged,
            physics_cg_relative_residual,
        ) = diagnostics

        if not bool(physics_cg_converged):
            raise RuntimeError(
                f"Linear solve did not converge at iteration {iteration}."
            )
        if not bool(mma_feasible):
            raise RuntimeError(
                f"MMA subproblem was infeasible at iteration {iteration}: "
                f"constraint={float(constraint_value):+.6e}, "
                f"bracketed={bool(mma_bracketed)}, "
                f"lambda={float(mma_lambda):.6e}, "
                f"phi={float(mma_phi):+.6e}"
            )

        final_change = float(change)
        history.append(
            {
                "iteration": int(iteration),
                "objective": float(objective_value),
                "normalized_objective": float(objective_value) / objective_scale,
                "constraint": float(constraint_value),
                "change": final_change,
                "iteration_time_s": float(iteration_time),
                "update_method": "MMA",
                "update_parameter": float(mma_lambda),
                "physics_cg_iterations": int(physics_cg_iterations),
                "physics_cg_relative_residual": float(
                    physics_cg_relative_residual
                ),
            }
        )

        plateau_reached = False
        if (
            iteration >= PLATEAU_MIN_ITER
            and len(history) > PLATEAU_WINDOW
        ):
            previous_compliance = history[-1 - PLATEAU_WINDOW]["objective"]
            current_compliance = history[-1]["objective"]
            final_plateau_relative_change = abs(
                current_compliance - previous_compliance
            ) / max(abs(current_compliance), np.finfo(float).tiny)
            plateau_reached = (
                final_plateau_relative_change <= PLATEAU_REL_TOL
            )

        should_print = (
            iteration == 1
            or iteration % PRINT_EVERY == 0
            or final_change <= CHANGE_TOL
            or plateau_reached
            or iteration == MAX_OPT_ITER
        )

        if should_print:
            print(
                f"iter {iteration:4d} | "
                f"f {float(objective_value):12.6f} | "
                f"rho {float(mean_design_rho):.6f} | "
                f"phys {float(mean_physical_rho):.6f} | "
                f"V/V* {float(volume_ratio):.6f} | "
                f"g {float(constraint_value):+.3e} | "
                f"Mnd {float(mnd):8.4f}% | "
                f"change {final_change:.6e} | "
                f"CG(K) {int(physics_cg_iterations):4d} | "
                f"time {iteration_time:.3f} s"
            )

        # ---------------------------------------------------------------------
        # Sigmund design-change criterion, supplemented by the same compliance
        # plateau criterion used for the Lazarov exposition cases.
        # ---------------------------------------------------------------------

        if final_change <= CHANGE_TOL:
            converged = True
            stopping_reason = "design_change"
            break

        if plateau_reached:
            converged = True
            stopping_reason = "compliance_plateau"
            break

    optimization_wall_time = time.perf_counter() - loop_start
    total_time = float(sum(iteration_times))

    # =========================================================================
    # Evaluate final design
    #
    # objective.value above belongs to rho BEFORE the final MMA update.
    # Evaluate once more at the actual final design.
    # =========================================================================

    final_evaluation_start = time.perf_counter()

    (
        solvers,
        final_objective,
        final_constraint,
        cg_state,
        final_linear_solve_diagnostics,
    ) = compiled_evaluation(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        cg_state,
    )

    jax.block_until_ready(
        (
            final_objective.value,
            final_constraint.value,
            cg_state,
            final_linear_solve_diagnostics.converged,
        )
    )

    if not bool(final_linear_solve_diagnostics.converged):
        raise RuntimeError("Linear solve did not converge for the final design.")

    final_evaluation_time = time.perf_counter() - final_evaluation_start

    # =========================================================================
    # Final physical density
    # =========================================================================

    final_rho_physical = conv_filter(rho, rho_mask)
    final_mean_design_rho = jnp.mean(rho)
    final_mean_rho = jnp.mean(final_rho_physical)
    final_volume_ratio = final_mean_rho / VOLFRAC
    final_mnd = non_discreteness(final_rho_physical)
    jax.block_until_ready(
        (
            final_rho_physical,
            final_mean_design_rho,
            final_mean_rho,
            final_volume_ratio,
            final_mnd,
        )
    )

    # =========================================================================
    # Literature comparison
    # =========================================================================

    compliance = float(final_objective.value)

    has_literature_reference = REFERENCE_COMPLIANCE is not None
    compliance_difference = (
        compliance - REFERENCE_COMPLIANCE
        if has_literature_reference
        else None
    )
    compliance_relative_difference = (
        compliance_difference / REFERENCE_COMPLIANCE
        if has_literature_reference
        else None
    )
    mnd_difference = (
        float(final_mnd) - REFERENCE_MND
        if REFERENCE_MND is not None
        else None
    )

    # =========================================================================
    # Final report
    # =========================================================================

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(f"converged:             {converged}")
    print(f"iterations:            {iteration}")
    print(f"stopping reason:       {stopping_reason}")
    print(f"final design change:   {final_change:.10e}")
    if final_plateau_relative_change is not None:
        print(
            "final compliance drift: "
            f"{final_plateau_relative_change:.10e} "
            f"over {PLATEAU_WINDOW} iterations"
        )
    print(f"setup time:            {setup_time:.4f} s")
    print(f"compilation time:      {compilation_time:.4f} s")
    print(f"initial evaluation:    {initial_evaluation_time:.4f} s")
    print(f"dummy execution:       {warmup_time:.4f} s")
    print(f"compile/warm-up time:  {compile_warmup_time:.4f} s")
    print(f"optimization time:     {total_time:.4f} s")
    print(f"optimization wall:     {optimization_wall_time:.4f} s")
    print(f"average iteration:     {total_time / iteration:.6f} s")
    print(f"final evaluation time: {final_evaluation_time:.4f} s")
    print()
    print(f"compliance:            {compliance:.10f}")
    if has_literature_reference:
        print(f"reference compliance:  {REFERENCE_COMPLIANCE:.10f}")
        print(f"absolute difference:   {compliance_difference:+.10f}")
        print(f"relative difference:   {compliance_relative_difference:+.6e}")
    else:
        print("reference compliance:  n/a (scaled exposition case)")
    print()
    print(f"mean design rho:       {float(final_mean_design_rho):.10f}")
    print(f"mean physical rho:     {float(final_mean_rho):.10f}")
    print(f"reference volume frac: {VOLFRAC:.10f}")
    print(f"normalized volume:     {float(final_volume_ratio):.10f}")
    print(f"constraint:            {float(final_constraint.value):+.6e}")
    print()
    print(f"Mnd:                   {float(final_mnd):.6f}%")
    if REFERENCE_MND is not None:
        print(f"reference Mnd:         {REFERENCE_MND:.6f}%")
        print(f"Mnd difference:        {mnd_difference:+.6f}%")
    print()
    print(f"TopaX iterations:      {iteration}")
    if REFERENCE_ITERATIONS is not None:
        print(f"reference iterations:  {REFERENCE_ITERATIONS}")
    print()
    print(
        f"design rho min/max:   {float(jnp.min(rho)):.6e} / {float(jnp.max(rho)):.6e}"
    )
    print(
        f"physical rho min/max: "
        f"{float(jnp.min(final_rho_physical)):.6e} / "
        f"{float(jnp.max(final_rho_physical)):.6e}"
    )

    # =========================================================================
    # Save result
    # =========================================================================

    output_start = time.perf_counter()
    save_densities(rho, final_rho_physical)
    save_history(history, args.runtime_csv)
    output_time = time.perf_counter() - output_start
    topax.solver.close_solver_state(solvers)

    case_wall_time = time.perf_counter() - case_wall_start
    physics_iterations = [
        row["physics_cg_iterations"] for row in history
    ]
    coarse_factor = 2 ** (N_LEVELS - 1)
    result = {
        "benchmark": "sigmund_2007",
        "case": case.label,
        "study_case": "mbb_density_filter",
        "literature_case": bool(has_literature_reference),
        "nx": int(ELEMENTS[0]),
        "ny": int(ELEMENTS[1]),
        "n_elements": int(np.prod(ELEMENTS)),
        "n_displacement_dofs": int(2 * np.prod(ELEMENTS + 1)),
        "n_levels": int(N_LEVELS),
        "galerkin_crossover_level": int(N_LEVELS - 1),
        "coarse_nx": int(ELEMENTS[0] // coarse_factor),
        "coarse_ny": int(ELEMENTS[1] // coarse_factor),
        "cg_tolerance": float(TOL),
        "cg_max_iterations": int(MAXITER),
        "optimization_method": "MMA",
        "optimization_max_iterations": int(MAX_OPT_ITER),
        "change_tolerance": float(CHANGE_TOL),
        "compliance_plateau_min_iterations": int(PLATEAU_MIN_ITER),
        "compliance_plateau_window": int(PLATEAU_WINDOW),
        "compliance_plateau_relative_tolerance": float(PLATEAU_REL_TOL),
        "filter_type": "linear density convolution",
        "filter_radius_elements": float(FILTER_RADIUS),
        "converged": bool(converged),
        "stopping_reason": stopping_reason,
        "iterations": int(iteration),
        "final_change": final_change,
        "final_compliance_plateau_relative_change": (
            final_plateau_relative_change
        ),
        "setup_time_s": float(setup_time),
        "compilation_time_s": float(compilation_time),
        "initial_evaluation_time_s": float(initial_evaluation_time),
        "warmup_time_s": float(warmup_time),
        "compile_warmup_time_s": float(compile_warmup_time),
        "optimization_time_s": float(total_time),
        "sum_iteration_time_s": float(total_time),
        "optimization_loop_wall_time_s": float(optimization_wall_time),
        "average_iteration_time_s": float(total_time / iteration),
        "final_evaluation_time_s": float(final_evaluation_time),
        "output_time_s": float(output_time),
        "case_wall_time_s": float(case_wall_time),
        "design_density_file": DESIGN_DENSITY_FILE.name,
        "physical_density_file": PHYSICAL_DENSITY_FILE.name,
        "history_file": "history.csv",
        "final_compliance": float(compliance),
        "reference_compliance": REFERENCE_COMPLIANCE,
        "compliance_absolute_difference": compliance_difference,
        "compliance_relative_error_percent": (
            100.0 * compliance_relative_difference
            if compliance_relative_difference is not None
            else None
        ),
        "mean_design_rho": float(final_mean_design_rho),
        "mean_physical_rho": float(final_mean_rho),
        "normalized_volume": float(final_volume_ratio),
        "constraint": float(final_constraint.value),
        "mnd_percent": float(final_mnd),
        "reference_mnd_percent": REFERENCE_MND,
        "mnd_difference_percent": mnd_difference,
        "reference_iterations": REFERENCE_ITERATIONS,
        "total_physics_cg_iterations": int(sum(physics_iterations)),
        "average_physics_cg_iterations": float(
            np.mean(physics_iterations)
        ),
        "maximum_physics_cg_iterations": int(max(physics_iterations)),
        "final_physics_cg_iterations": int(
            final_linear_solve_diagnostics.physics["n_iter"]
        ),
        "final_physics_cg_relative_residual": float(
            final_linear_solve_diagnostics.physics["relative_residual"]
        ),
    }
    result.update(runtime_metadata())
    save_summary(result)
    return result


if __name__ == "__main__":
    main()
