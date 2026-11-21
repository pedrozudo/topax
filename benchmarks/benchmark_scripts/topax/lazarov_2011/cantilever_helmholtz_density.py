import argparse
import csv
import gc
import json
import os
import platform
import resource
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

import topax

from topax import (
    FORMULATION_MMA,
    PROBLEM_COMPLIANCE,
)


# =============================================================================
# JAX
# =============================================================================

jax.config.update(
    "jax_enable_x64",
    True,
)


# =============================================================================
# Lazarov & Sigmund (2011)
#
# "Filters in topology optimization based on Helmholtz-type
#  differential equations"
#
# Figure 8:
#
#     Helmholtz PDE density filtering
#     mesh independence
#     r = 0.02
#
# Paper orientation:
#
#     domain: 1 x 4
#     meshes:
#
#         100 x 400
#         200 x 800
#         300 x 1200
#
# TopaX orientation used here:
#
#     domain: 4 x 1
#     ELEMENTS = (nx, ny)
#
#     (400, 100)
#     (800, 200)
#     (1200, 300)
#
# Figure-8 reference compliances:
#
#     100 x 400:   16.6556
#     200 x 800:   16.6690
#     300 x 1200:  16.6785
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

    @property
    def elements(self):
        return np.array(
            [
                self.nx,
                self.ny,
            ],
            dtype=np.int64,
        )

    @property
    def topax_label(self):
        return f"{self.nx}x{self.ny}"

    @property
    def paper_label(self):
        return f"{self.ny}x{self.nx}"

    @property
    def tag(self):
        return f"{self.nx:04d}x{self.ny:04d}"


CASES = (
    MeshCase(
        nx=400,
        ny=100,
        n_levels=2,
        reference_compliance=16.6556,
    ),
    MeshCase(
        nx=800,
        ny=200,
        n_levels=2,
        reference_compliance=16.6690,
    ),
    MeshCase(
        nx=1200,
        ny=300,
        n_levels=2,
        reference_compliance=16.6785,
    ),
    MeshCase(nx=1600, ny=400, n_levels=3),
    MeshCase(nx=3200, ny=800, n_levels=3),
    MeshCase(nx=6400, ny=1600, n_levels=3),
)

LITERATURE_CASES = tuple(
    case for case in CASES if case.reference_compliance is not None
)


# =============================================================================
# Physical problem
# =============================================================================

SIZES = np.array(
    [
        4.0,
        1.0,
    ],
    dtype=np.float64,
)

ELASTIC_MODULE = 1.0

POISSON_RATIO = 0.2

THICKNESS = 0.01

SIMP_PENALTY = 3

VOLFRAC = 0.5

LOAD = -0.2


# =============================================================================
# Density lower bounds
#
# The design variables are explicitly bounded away from zero.
#
# For p = 3:
#
#     (1e-3)^3 = 1e-9
#
# so DESIGN_RHO_MIN and the modified-SIMP stiffness floor correspond
# naturally.
# =============================================================================

DESIGN_RHO_MIN = 1.0e-3

RHO_MIN = 1.0e-9


# =============================================================================
# Helmholtz filter
#
# Lazarov PDE:
#
#     -r^2 Laplacian(rho_tilde) + rho_tilde = rho
#
# Paper:
#
#     r = 0.02
#
# TopaX public radius convention:
#
#     r_h = 2 sqrt(3) r
#
# construct_local_helmholtz() converts internally according to
#
#     r = r_h / (2 sqrt(3)).
# =============================================================================

PAPER_HELMHOLTZ_R = 0.02

FILTER_RADIUS = 2.0 * np.sqrt(3.0) * PAPER_HELMHOLTZ_R


# =============================================================================
# Multigrid / CG
# =============================================================================

# The hierarchy depth is part of each mesh case. The scaled experiments are
# capped at four levels.

TOL = 1.0e-10

MAXITER = 500

DAMPING = 0.8


# =============================================================================
# Optimality Criteria
# =============================================================================

OC_MOVE = 0.2

OC_BISECT_STEPS = 64

OC_PARAMETERS = topax.optimizer.oc.init_oc_params(
    xmin=DESIGN_RHO_MIN,
    move=OC_MOVE,
    max_bisect=OC_BISECT_STEPS,
    bisect_tol=0.0,
)

OC_STEP = topax.optimizer.oc.construct_oc_step(OC_PARAMETERS)

# Paper stopping criterion:
#
#     max_i |rho_i^(k+1) - rho_i^k| < 0.01

CHANGE_TOL = 0.01

# Fine meshes may need a large number of OC iterations.
MAX_OPT_ITER = 10_000

# A run may also stop once compliance has changed by less than 0.001% over
# 100 iterations.
PLATEAU_MIN_ITER = 500
PLATEAU_WINDOW = 100
PLATEAU_REL_TOL = 1.0e-5


# =============================================================================
# Console output
#
# Printing every single iteration becomes excessive for the 3-mesh study.
# We always print:
#
#     iteration 1
#     every PRINT_EVERY iterations
#     the converged iteration
#     the final allowed iteration
# =============================================================================

PRINT_EVERY = 10


# =============================================================================
# Optional gradient check
#
# Recommended while validating the implementation, but disable for the
# normal three-mesh run because each finite-difference check requires
# additional full PDE solves.
# =============================================================================

RUN_GRADIENT_CHECK = False

GRADIENT_CHECK_EPS = 1.0e-6

GRADIENT_CHECK_RTOL = 1.0e-4


# =============================================================================
# Benchmark acceptance criterion
#
# This is OUR benchmark tolerance, not a criterion stated in the paper.
# =============================================================================

REFERENCE_RELATIVE_TOL = 0.01


# =============================================================================
# Output
# =============================================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "benchmarks/results/lazarov_2011/figure_8_mesh_independence"
)

SUMMARY_CSV = OUTPUT_DIR / "summary.csv"

SUMMARY_JSON = OUTPUT_DIR / "summary.json"


def configure_output_dir(output_dir):
    global OUTPUT_DIR
    global SUMMARY_CSV
    global SUMMARY_JSON

    OUTPUT_DIR = Path(output_dir).expanduser().resolve()
    SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
    SUMMARY_JSON = OUTPUT_DIR / "summary.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Lazarov & Sigmund (2011) Figure 8 validation benchmark."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX platform: CPU or NVIDIA CUDA GPU",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.topax_label for case in CASES),
        help=(
            "Topax mesh to run; repeat for multiple cases "
            "(default: the three Figure 8 cases)"
        ),
    )
    parser.add_argument(
        "--n-levels",
        type=int,
        choices=(2, 3, 4),
        help="Override the selected cases' default multigrid hierarchy depth",
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
        help="write one selected case's iteration history to this CSV file",
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


# =============================================================================
# Helpers
# =============================================================================


def identity_projection(x):
    """
    Figure 8 is the Helmholtz density-filter case without a black/white
    projection.
    """

    return x


def non_discreteness(rho):
    """
    Grayness diagnostic only.

    Lazarov & Sigmund do not provide an Mnd target for Figure 8.
    """

    return 100.0 * jnp.mean(4.0 * rho * (1.0 - rho))


def paper_comparison_compliance(
    raw_compliance,
):
    """
    Compliance quantity used to compare this 2D TopaX model against the
    Figure-8 values.

    TopaX's explicit 2D thickness scaling gives approximately

        C_raw ~ C_paper / thickness

    for thickness = 0.01.

    Keep this separate from objective normalization C/C0.
    """

    return THICKNESS * raw_compliance


def case_output_paths(
    case,
):
    prefix = OUTPUT_DIR / f"cantilever_r002_{case.tag}"

    return {
        "rho": Path(f"{prefix}_rho.npy"),
        "rho_physical": Path(f"{prefix}_rho_physical.npy"),
        "history": Path(f"{prefix}_history.csv"),
    }


# =============================================================================
# Boundary-condition diagnostics
# =============================================================================


def print_restriction_diagnostics(
    case,
    restrictions,
):
    print()
    print("=" * 80)
    print("RESTRICTION CHECK")
    print("=" * 80)

    for level_index, restriction in enumerate(restrictions):
        restriction.block_until_ready()

        n_free = int(jnp.count_nonzero(restriction))

        n_total = int(restriction.size)

        n_restricted = n_total - n_free

        print(
            f"level {level_index}: "
            f"shape={restriction.shape}, "
            f"free={n_free}, "
            f"restricted={n_restricted}"
        )

    # Full left boundary:
    #
    #     ny + 1 nodes
    #     2 displacement components per node

    fine = restrictions[0]

    n_restricted = fine.size - int(jnp.count_nonzero(fine))

    expected = 2 * (case.ny + 1)

    print()

    print(f"fine restricted DOFs:  {n_restricted}")

    print(f"expected:              {expected}")

    if n_restricted != expected:
        raise RuntimeError(
            "Cantilever restriction count is incorrect. "
            f"Expected {expected}, got {n_restricted}."
        )

    print("restriction count:     PASS")


def print_force_diagnostics(
    case,
    forces,
    restrictions,
):
    forces.block_until_ready()

    force_norm = jnp.linalg.norm(forces)

    force_abs_sum = jnp.sum(jnp.abs(forces))

    force_nonzero = jnp.count_nonzero(forces)

    force_indices = jnp.argwhere(forces != 0)

    active_forces = forces * restrictions[0]

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

    expected_index = np.array(
        [
            1,
            case.nx,
            case.ny // 2,
        ],
        dtype=np.int64,
    )

    if int(force_nonzero) != 1:
        raise RuntimeError("Expected exactly one nonzero force entry.")

    actual_index = np.asarray(force_indices[0])

    if not np.array_equal(
        actual_index,
        expected_index,
    ):
        raise RuntimeError(
            "Point load is at the wrong node. "
            f"Expected {expected_index}, "
            f"got {actual_index}."
        )

    if not np.isclose(
        float(force_norm),
        abs(LOAD),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("Force norm does not match the paper load.")

    if not np.isclose(
        float(active_force_norm),
        abs(LOAD),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("The point load is removed by the Dirichlet restrictions.")

    print()

    print(f"norm after restriction: {float(active_force_norm):.16e}")

    print()

    print("force check:           PASS")


# =============================================================================
# Standalone Helmholtz density filter
#
# Used for:
#
#     1. constant-field diagnostic
#     2. final physical density
#
# The optimization itself uses TopaX's production FILTER_HELMHOLTZ path.
# =============================================================================


def construct_density_filter(
    case,
    grid,
    helmholtz,
    local_helmholtz_operator,
):
    helmholtz_mask = jnp.ones(
        case.elements,
        dtype=jnp.float64,
    )

    level = grid.fine_level

    def helmholtz_matvec(value):
        return topax.operator.matrix_free_matvec(
            level,
            value,
            local_helmholtz_operator,
            helmholtz_mask,
        )

    diagonal = topax.operator.matrix_free_diagonal(
        level,
        local_helmholtz_operator,
        helmholtz_mask,
    )

    def filter_density(
        rho,
    ):
        rho_nodal = topax.helmholtz_to_nodal(
            grid.fine_level,
            helmholtz,
            rho,
        )

        (
            rho_nodal_filtered,
            info,
        ) = topax.cg.cg_solve(
            helmholtz_matvec,
            rho_nodal,
            tol=TOL,
            maxiter=MAXITER,
            M=lambda x: x / diagonal,
        )

        rho_physical = topax.helmholtz_to_elemental(
            grid.fine_level,
            rho_nodal_filtered,
        )

        rho_physical = rho_physical.reshape(
            (
                case.nx,
                case.ny,
            )
        )

        return (
            rho_physical,
            info,
        )

    return jax.jit(filter_density)


def print_filter_diagnostics(
    filter_density,
    rho,
):
    (
        rho_physical,
        info,
    ) = filter_density(rho)

    jax.block_until_ready(
        (
            rho_physical,
            info,
        )
    )

    if not bool(info["converged"]):
        raise RuntimeError("Standalone Helmholtz filter did not converge.")

    design_mean = jnp.mean(rho)

    physical_mean = jnp.mean(rho_physical)

    mean_difference = physical_mean - design_mean

    max_abs_difference = jnp.max(jnp.abs(rho_physical - rho))

    jax.block_until_ready(
        (
            design_mean,
            physical_mean,
            mean_difference,
            max_abs_difference,
        )
    )

    print()
    print("=" * 80)
    print("HELMHOLTZ FILTER CHECK")
    print("=" * 80)

    print(f"paper PDE r:           {PAPER_HELMHOLTZ_R:.16e}")

    print(f"TopaX filter radius:   {FILTER_RADIUS:.16e}")

    print()

    print(
        f"design min/max:        "
        f"{float(jnp.min(rho)):.16e} / "
        f"{float(jnp.max(rho)):.16e}"
    )

    print(
        f"physical min/max:      "
        f"{float(jnp.min(rho_physical)):.16e} / "
        f"{float(jnp.max(rho_physical)):.16e}"
    )

    print(f"design mean:           {float(design_mean):.16e}")

    print(f"physical mean:         {float(physical_mean):.16e}")

    print(f"physical-design mean:  {float(mean_difference):+.16e}")

    print(f"max |physical-design|: {float(max_abs_difference):.16e}")

    if float(max_abs_difference) > 1.0e-9:
        raise RuntimeError("Uniform density is not preserved by the Helmholtz filter.")

    print()

    print("filter check:          PASS")


# =============================================================================
# Initial objective diagnostics
# =============================================================================


def print_initial_value_and_grad_diagnostics(
    objective,
    constraint,
    objective_scale,
):
    objective_grad_norm = jnp.linalg.norm(objective.grad)

    normalized_grad_norm = jnp.linalg.norm(objective.grad / objective_scale)

    constraint_grad_norm = jnp.linalg.norm(constraint.grad)

    jax.block_until_ready(
        (
            objective.value,
            objective.grad,
            constraint.value,
            constraint.grad,
            objective_grad_norm,
            normalized_grad_norm,
            constraint_grad_norm,
        )
    )

    raw = float(objective.value)

    normalized = raw / objective_scale

    comparison = paper_comparison_compliance(raw)

    print()
    print("=" * 80)
    print("INITIAL VALUE + GRAD CHECK")
    print("=" * 80)

    print(f"raw compliance:        {raw:.16e}")

    print(f"objective scale C0:    {objective_scale:.16e}")

    print(f"normalized C/C0:       {normalized:.16e}")

    print(f"comparison t*C:        {comparison:.16e}")

    print()

    print(f"raw grad min:          {float(jnp.min(objective.grad)):.16e}")

    print(f"raw grad max:          {float(jnp.max(objective.grad)):.16e}")

    print(f"raw grad norm:         {float(objective_grad_norm):.16e}")

    print(f"normalized grad norm:  {float(normalized_grad_norm):.16e}")

    print()

    print(f"constraint:            {float(constraint.value):+.16e}")

    print(f"constraint grad norm:  {float(constraint_grad_norm):.16e}")

    if not np.isfinite(raw):
        raise RuntimeError("Initial compliance is not finite.")

    if raw <= 0.0:
        raise RuntimeError("Initial compliance is zero or negative.")

    if not np.isfinite(float(objective_grad_norm)):
        raise RuntimeError("Initial compliance gradient is not finite.")

    if float(objective_grad_norm) == 0.0:
        raise RuntimeError("Initial compliance gradient is zero.")

    print()

    print("initial value+grad:    PASS")


# =============================================================================
# Optional finite-difference check
# =============================================================================


def run_gradient_check(
    value_and_grad,
    solvers,
    rho,
    forces,
    rho_mask,
    restrictions,
    initial_cg_state,
):
    print()
    print("=" * 80)
    print("DIRECTIONAL GRADIENT CHECK")
    print("=" * 80)

    key = jax.random.PRNGKey(1234)

    direction = jax.random.normal(
        key,
        rho.shape,
        dtype=rho.dtype,
    )

    # Remove the constant/volume component.
    direction = direction - jnp.mean(direction)

    direction = direction / jnp.linalg.norm(direction)

    eps = jnp.asarray(
        GRADIENT_CHECK_EPS,
        dtype=rho.dtype,
    )

    rho_plus = jnp.clip(
        rho + eps * direction,
        DESIGN_RHO_MIN,
        1.0,
    )

    rho_minus = jnp.clip(
        rho - eps * direction,
        DESIGN_RHO_MIN,
        1.0,
    )

    (
        solvers,
        objective_0,
        _,
        _,
        diagnostics_0,
    ) = value_and_grad(
        solvers,
        rho,
        forces,
        rho_mask,
        restrictions,
        initial_cg_state,
    )

    (
        solvers,
        objective_plus,
        _,
        _,
        diagnostics_plus,
    ) = value_and_grad(
        solvers,
        rho_plus,
        forces,
        rho_mask,
        restrictions,
        initial_cg_state,
    )

    (
        solvers,
        objective_minus,
        _,
        _,
        diagnostics_minus,
    ) = value_and_grad(
        solvers,
        rho_minus,
        forces,
        rho_mask,
        restrictions,
        initial_cg_state,
    )

    jax.block_until_ready(
        (
            objective_0.value,
            objective_0.grad,
            objective_plus.value,
            objective_minus.value,
            diagnostics_0.converged,
            diagnostics_plus.converged,
            diagnostics_minus.converged,
        )
    )

    if not all(
        bool(diagnostics.converged)
        for diagnostics in (diagnostics_0, diagnostics_plus, diagnostics_minus)
    ):
        raise RuntimeError("A linear solve failed during the gradient check.")

    finite_difference = (objective_plus.value - objective_minus.value) / (2.0 * eps)

    adjoint = jnp.vdot(
        objective_0.grad,
        direction,
    )

    absolute_error = jnp.abs(finite_difference - adjoint)

    relative_error = absolute_error / jnp.maximum(
        jnp.abs(finite_difference),
        1.0e-30,
    )

    jax.block_until_ready(
        (
            finite_difference,
            adjoint,
            absolute_error,
            relative_error,
        )
    )

    print(f"epsilon:               {float(eps):.6e}")

    print(f"finite difference:     {float(finite_difference):+.16e}")

    print(f"adjoint:               {float(adjoint):+.16e}")

    print(f"absolute error:        {float(absolute_error):.16e}")

    print(f"relative error:        {float(relative_error):.16e}")

    if float(relative_error) > GRADIENT_CHECK_RTOL:
        raise RuntimeError("Directional compliance gradient check failed.")

    print()

    print("gradient check:        PASS")

    return solvers


# =============================================================================
# Optimality Criteria
#
# The gradient supplied to OC is
#
#     d(C/C0)/drho
#
# while TopaX itself continues to return the physical/raw compliance.
#
# The normalized volume constraint is
#
#     mean(rho_physical) / VOLFRAC - 1
#
# in the production value-and-grad path.
#
# Because the homogeneous-Neumann Helmholtz filter preserves volume, the OC
# bisection can enforce the equivalent condition
#
#     mean(rho_design) = VOLFRAC.
# =============================================================================


# =============================================================================
# Save one mesh case
# =============================================================================


def save_case_results(
    case,
    rho,
    rho_physical,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = case_output_paths(case)

    rho_np = np.asarray(rho)

    rho_physical_np = np.asarray(rho_physical)

    np.save(
        paths["rho"],
        rho_np,
    )

    np.save(
        paths["rho_physical"],
        rho_physical_np,
    )

    print()

    print(f"saved design density:   {paths['rho']}")

    print(f"saved physical density: {paths['rho_physical']}")


def save_case_history(case, history, path=None):
    path = case_output_paths(case)["history"] if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"saved iteration history: {path}")


# =============================================================================
# Construct one mesh case
# =============================================================================


def construct_problem(
    case,
):
    elements = case.elements

    # =========================================================================
    # Physical models
    # =========================================================================

    elastic = topax.construct_elastic(
        ELASTIC_MODULE,
        POISSON_RATIO,
        THICKNESS,
    )

    helmholtz = topax.construct_helmholtz(
        FILTER_RADIUS,
        THICKNESS,
    )

    # =========================================================================
    # Geometry
    # =========================================================================

    grid = topax.construct_grid(
        case.n_levels,
        SIZES,
        elements,
    )

    # =========================================================================
    # Local operators
    # =========================================================================

    local_helmholtz_operators = tuple(
        topax.construct_local_helmholtz(
            level,
            helmholtz,
        )
        for level in grid.levels
    )

    fine_local_stiffness = topax.construct_local_stiffness(
        grid.fine_level,
        elastic,
    )

    # =========================================================================
    # Cantilever BC
    #
    # Full left edge fixed:
    #
    #     ux = 0
    #     uy = 0
    #
    # TopaX selectors are normalized to [0, 1]^2.
    # =========================================================================

    dirichlet_boundary_conditions = [
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
    ]

    restrictions = topax.construct_restrictions(
        grid,
        elastic,
        dirichlet_boundary_conditions,
    )

    # =========================================================================
    # Point load
    #
    # Downward 0.2 load at the midpoint of the right edge.
    # =========================================================================

    concentrated_forces = [
        [],
        [
            {
                "center": [
                    1.0,
                    0.5,
                ],
                "size": [
                    0.0,
                    0.0,
                ],
                "intensity": LOAD,
            },
        ],
    ]

    forces = topax.construct_forces(
        grid,
        elastic,
        concentrated_forces,
    )

    print_restriction_diagnostics(
        case,
        restrictions,
    )

    print_force_diagnostics(
        case,
        forces,
        restrictions,
    )

    # =========================================================================
    # Coarse sparse structures
    # =========================================================================

    start = time.perf_counter()

    coarse_physics_structure = topax.sparse.construct_sparse_physics_structure(
        grid.coarse_level,
        n_dofs_node=grid.n_dims,
        restriction=restrictions[-1],
    )

    coarse_helmholtz_structure = (
        topax.sparse.construct_sparse_helmholtz_structure(
            grid.coarse_level,
            local_helmholtz_operators[-1],
        )
    )

    jax.effects_barrier()

    elapsed = time.perf_counter() - start

    print()

    print(f"coarse physics structure: {1000.0 * elapsed:.4f} ms")

    # =========================================================================
    # Standalone diagnostic filter
    # =========================================================================

    filter_density = construct_density_filter(
        case,
        grid,
        helmholtz,
        local_helmholtz_operators[0],
    )

    return (
        helmholtz,
        grid,
        local_helmholtz_operators,
        fine_local_stiffness,
        restrictions,
        forces,
        coarse_physics_structure,
        coarse_helmholtz_structure,
        filter_density,
    )


# =============================================================================
# Run one Figure-8 mesh
# =============================================================================


def run_case(
    case,
    runtime_csv=None,
):
    case_wall_start = time.perf_counter()

    print()
    print()
    print("#" * 100)
    print("#" * 100)
    print(
        f"FIGURE 8 CASE: paper mesh {case.paper_label} | TopaX mesh {case.topax_label}"
    )
    print("#" * 100)
    print("#" * 100)

    print()
    print(f"elements:              {tuple(case.elements)}")

    print(f"sizes:                 {tuple(SIZES)}")

    print(f"E:                     {ELASTIC_MODULE}")

    print(f"Poisson ratio:         {POISSON_RATIO}")

    print(f"thickness:             {THICKNESS}")

    print(f"SIMP penalty:          {SIMP_PENALTY}")

    print(f"design rho min:        {DESIGN_RHO_MIN}")

    print(f"TopaX stiffness floor: {RHO_MIN}")

    print(f"volume fraction:       {VOLFRAC}")

    print(f"load:                  {LOAD}")

    print(f"paper Helmholtz r:     {PAPER_HELMHOLTZ_R}")

    print(f"TopaX filter radius:   {FILTER_RADIUS:.16e}")

    if case.reference_compliance is not None:
        print(f"paper reference C:     {case.reference_compliance:.10f}")
    else:
        print("paper reference C:     n/a (scaled exposition case)")

    print(f"OC move:               {OC_MOVE}")

    print(f"change tolerance:      {CHANGE_TOL}")

    print(
        "compliance plateau:    "
        f"{PLATEAU_REL_TOL:.1e} over {PLATEAU_WINDOW} iterations "
        f"after iteration {PLATEAU_MIN_ITER}"
    )

    print(f"maximum iterations:    {MAX_OPT_ITER}")

    print(f"MG levels:             {case.n_levels}")

    print(f"CG tolerance:          {TOL}")

    print(f"device:                {jax.devices()[0]}")

    print(f"x64 enabled:           {jax.config.jax_enable_x64}")

    # =========================================================================
    # Build problem
    # =========================================================================

    setup_start = time.perf_counter()

    (
        helmholtz,
        grid,
        local_helmholtz_operators,
        fine_local_stiffness,
        restrictions,
        forces,
        coarse_physics_structure,
        coarse_helmholtz_structure,
        filter_density,
    ) = construct_problem(case)

    solvers = topax.solver.setup_cg_filter_solver_state(
        coarse_physics_structure,
        coarse_helmholtz_structure,
        grid,
        restrictions,
    )

    # =========================================================================
    # Initial design
    # =========================================================================

    rho = VOLFRAC * jnp.ones(
        (
            case.nx,
            case.ny,
        ),
        dtype=jnp.float64,
    )

    rho_mask = jnp.ones_like(
        rho,
        dtype=jnp.float64,
    )

    # =========================================================================
    # Uniform Helmholtz test
    # =========================================================================

    print_filter_diagnostics(
        filter_density,
        rho,
    )

    # =========================================================================
    # Production value + gradient
    #
    # IMPORTANT:
    #
    # normalized_volume_constraint is evaluated on rho_projected by your
    # registered FILTER_HELMHOLTZ implementation:
    #
    #     g =
    #         mean(rho_projected) / VOLFRAC
    #         - 1
    #
    # The corresponding gradient is then propagated backward through the
    # Helmholtz solve.
    # =========================================================================

    (
        value_and_grad,
        initial_cg_state,
    ) = topax.construct_value_and_grad(
        problem=PROBLEM_COMPLIANCE,
        filtering=topax.FILTER_HELMHOLTZ,
        formulation=FORMULATION_MMA,
        constraint_function=topax.normalized_volume_constraint,
        helmholtz=helmholtz,
        local_operators_helmholtz=local_helmholtz_operators,
        fine_local_operator=fine_local_stiffness,
        grid=grid,
        coarse_physics_structure=coarse_physics_structure,
        coarse_helmholtz_structure=coarse_helmholtz_structure,
        projection=identity_projection,
        simp_penalty=SIMP_PENALTY,
        rho_min=RHO_MIN,
        tol=TOL,
        maxiter=MAXITER,
        damping=DAMPING,
        volfrac=VOLFRAC,
    )

    # Filtering CG state contains aliased leaves.
    # Do not donate the whole state.

    value_and_grad = jax.jit(value_and_grad)

    # =========================================================================
    # Optional gradient validation
    # =========================================================================

    if RUN_GRADIENT_CHECK:
        solvers = run_gradient_check(
            value_and_grad,
            solvers,
            rho,
            forces,
            rho_mask,
            restrictions,
            initial_cg_state,
        )

    setup_time = time.perf_counter() - setup_start

    # =========================================================================
    # Explicit evaluation compilation and untimed warm-up
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
        raise RuntimeError("A linear solve failed during initial evaluation.")

    initial_evaluation_time = time.perf_counter() - initial_evaluation_start

    objective_scale = float(initial_objective.value)

    if not np.isfinite(objective_scale):
        raise RuntimeError("Initial compliance is not finite.")

    if objective_scale <= 0.0:
        raise RuntimeError("Initial compliance must be positive.")

    # Match the Träff/Wang protocol: retain the native solver resources from
    # the untimed evaluation, but start the timed numerical state from the
    # original warm start.
    cg_state = initial_cg_state
    del evaluated_cg_state

    print_initial_value_and_grad_diagnostics(
        initial_objective,
        initial_constraint,
        objective_scale,
    )

    # =========================================================================
    # Complete compiled optimization step
    # =========================================================================

    def optimization_step(solvers, rho, cg_state):
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
        normalized_objective = objective.value / objective_scale
        rho_new, oc_info = OC_STEP(
            rho,
            objective.grad / objective_scale,
            jnp.sum(rho) - VOLFRAC * rho.size,
            jnp.ones_like(rho),
        )
        change = oc_info["change"]
        oc_lambda = oc_info["lam"]
        physics = linear_solve_diagnostics.physics
        helmholtz_forward = linear_solve_diagnostics.helmholtz_forward
        helmholtz_objective = (
            linear_solve_diagnostics.helmholtz_objective_adjoint
        )
        helmholtz_constraint = (
            linear_solve_diagnostics.helmholtz_constraint_adjoint
        )
        diagnostics = (
            objective.value,
            normalized_objective,
            THICKNESS * objective.value,
            constraint.value,
            change,
            jnp.mean(rho),
            jnp.mean(rho) / VOLFRAC,
            oc_lambda,
            physics["n_iter"],
            physics["converged"],
            physics["relative_residual"],
            helmholtz_forward["n_iter"],
            helmholtz_forward["converged"],
            helmholtz_forward["relative_residual"],
            helmholtz_objective["n_iter"],
            helmholtz_objective["converged"],
            helmholtz_objective["relative_residual"],
            helmholtz_constraint["n_iter"],
            helmholtz_constraint["converged"],
            helmholtz_constraint["relative_residual"],
        )
        return solvers, rho_new, cg_state, diagnostics

    optimization_step = jax.jit(optimization_step)

    print("Compiling complete optimization step...", flush=True)
    compilation_start = time.perf_counter()
    compiled_step = optimization_step.lower(
        solvers,
        rho,
        cg_state,
    ).compile()
    step_compilation_time = time.perf_counter() - compilation_start
    compilation_time += step_compilation_time
    print(f"Optimization step compiled in {step_compilation_time:.3f} s")

    warmup_rho = jnp.array(rho, copy=True)
    warmup_cg_state = jax.tree.map(
        lambda value: jnp.array(value, copy=True),
        cg_state,
    )
    print("Running untimed dummy iteration...", flush=True)
    warmup_start = time.perf_counter()
    warmup_outputs = compiled_step(
        solvers,
        warmup_rho,
        warmup_cg_state,
    )
    jax.block_until_ready(warmup_outputs)
    solvers = warmup_outputs[0]
    warmup_diagnostics = warmup_outputs[-1]
    warmup_time = time.perf_counter() - warmup_start
    print(f"Dummy iteration completed in {warmup_time:.3f} s")
    for name, converged_index in (
        ("physics", 9),
        ("Helmholtz forward", 12),
        ("Helmholtz objective adjoint", 15),
        ("Helmholtz constraint adjoint", 18),
    ):
        if not bool(warmup_diagnostics[converged_index]):
            raise RuntimeError(f"{name} solve failed during dummy execution.")
    del warmup_outputs, warmup_rho, warmup_cg_state

    compile_warmup_time = (
        compilation_time + initial_evaluation_time + warmup_time
    )

    # =========================================================================
    # Optimization
    # =========================================================================

    converged = False

    final_change = np.inf
    final_plateau_relative_change = None
    stopping_reason = "iteration_limit"

    history = []
    iteration_times = []

    print()
    print("=" * 100)
    print("OPTIMIZATION")
    print("=" * 100)

    loop_start = time.perf_counter()

    for iteration in range(
        1,
        MAX_OPT_ITER + 1,
    ):
        iteration_start = time.perf_counter()
        solvers, rho, cg_state, diagnostics = compiled_step(
            solvers,
            rho,
            cg_state,
        )
        jax.block_until_ready((solvers, rho, cg_state, diagnostics))
        iteration_time = time.perf_counter() - iteration_start
        iteration_times.append(iteration_time)

        (
            raw_compliance,
            normalized_objective,
            comparison_compliance,
            constraint_value,
            change,
            current_mean_rho,
            current_volume_ratio,
            oc_lambda,
            physics_cg_iterations,
            physics_cg_converged,
            physics_cg_relative_residual,
            helmholtz_forward_cg_iterations,
            helmholtz_forward_cg_converged,
            helmholtz_forward_cg_relative_residual,
            helmholtz_objective_cg_iterations,
            helmholtz_objective_cg_converged,
            helmholtz_objective_cg_relative_residual,
            helmholtz_constraint_cg_iterations,
            helmholtz_constraint_cg_converged,
            helmholtz_constraint_cg_relative_residual,
        ) = diagnostics

        for name, solve_converged in (
            ("physics", physics_cg_converged),
            ("Helmholtz forward", helmholtz_forward_cg_converged),
            (
                "Helmholtz objective adjoint",
                helmholtz_objective_cg_converged,
            ),
            (
                "Helmholtz constraint adjoint",
                helmholtz_constraint_cg_converged,
            ),
        ):
            if not bool(solve_converged):
                raise RuntimeError(
                    f"{name} solve failed at iteration {iteration}."
                )

        final_change = float(change)

        history_row = {
            "iteration": int(iteration),
            "objective": float(raw_compliance),
            "normalized_objective": float(normalized_objective),
            "comparison_compliance": float(comparison_compliance),
            "constraint": float(constraint_value),
            "change": float(final_change),
            "iteration_time_s": float(iteration_time),
            "update_method": "OC",
            "update_parameter": float(oc_lambda),
            "physics_cg_iterations": int(physics_cg_iterations),
            "physics_cg_relative_residual": float(
                physics_cg_relative_residual
            ),
        }
        for name, n_iter, relative_residual in (
            (
                "helmholtz_forward",
                helmholtz_forward_cg_iterations,
                helmholtz_forward_cg_relative_residual,
            ),
            (
                "helmholtz_objective_adjoint",
                helmholtz_objective_cg_iterations,
                helmholtz_objective_cg_relative_residual,
            ),
            (
                "helmholtz_constraint_adjoint",
                helmholtz_constraint_cg_iterations,
                helmholtz_constraint_cg_relative_residual,
            ),
        ):
            history_row[f"{name}_cg_iterations"] = int(n_iter)
            history_row[f"{name}_cg_relative_residual"] = float(
                relative_residual
            )
        history.append(history_row)

        plateau_reached = False
        if (
            iteration >= PLATEAU_MIN_ITER
            and len(history) > PLATEAU_WINDOW
        ):
            previous_compliance = history[-1 - PLATEAU_WINDOW][
                "comparison_compliance"
            ]
            current_compliance = history[-1]["comparison_compliance"]
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
                f"C {raw_compliance:12.6f} | "
                f"C/C0 {normalized_objective:9.6f} | "
                f"t*C {comparison_compliance:9.6f} | "
                f"vol {float(current_mean_rho):.6f} | "
                f"V/V* {float(current_volume_ratio):.6f} | "
                f"g {float(constraint_value):+.3e} | "
                f"change {final_change:.6e} | "
                f"lambda {float(oc_lambda):.6e} | "
                f"CG(K) {int(physics_cg_iterations):4d} | "
                f"time {iteration_time:.3f} s"
            )

        if final_change <= CHANGE_TOL:
            converged = True
            stopping_reason = "design_change"
            break

        if plateau_reached:
            converged = True
            stopping_reason = "compliance_plateau"
            break

    optimization_wall_time = time.perf_counter() - loop_start
    optimization_time = float(sum(iteration_times))

    # =========================================================================
    # Final objective evaluated at the UPDATED final rho
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
            final_objective.grad,
            final_constraint.value,
            final_constraint.grad,
            cg_state,
            final_linear_solve_diagnostics.converged,
        )
    )

    if not bool(final_linear_solve_diagnostics.converged):
        raise RuntimeError("A linear solve failed for the final design.")

    final_evaluation_time = time.perf_counter() - final_evaluation_start

    # =========================================================================
    # Final physical density
    # =========================================================================

    final_rho_physical = topax.helmholtz_to_elemental(
        grid.fine_level,
        cg_state.helmholtz_value0,
    ).reshape((case.nx, case.ny))

    jax.block_until_ready(final_rho_physical)

    final_mean_design_rho = jnp.mean(rho)

    final_mean_physical_rho = jnp.mean(final_rho_physical)

    final_normalized_volume = final_mean_physical_rho / VOLFRAC

    final_filter_volume_difference = final_mean_physical_rho - final_mean_design_rho

    final_mnd = non_discreteness(final_rho_physical)

    jax.block_until_ready(
        (
            final_mean_design_rho,
            final_mean_physical_rho,
            final_normalized_volume,
            final_filter_volume_difference,
            final_mnd,
        )
    )

    # =========================================================================
    # Compliance comparison
    # =========================================================================

    raw_compliance = float(final_objective.value)

    normalized_compliance = raw_compliance / objective_scale

    comparison_compliance = paper_comparison_compliance(raw_compliance)

    has_literature_reference = case.reference_compliance is not None
    absolute_difference = (
        comparison_compliance - case.reference_compliance
        if has_literature_reference
        else None
    )
    relative_difference = (
        absolute_difference / case.reference_compliance
        if has_literature_reference
        else None
    )
    relative_error_percent = (
        100.0 * relative_difference
        if has_literature_reference
        else None
    )
    compliance_pass = (
        abs(relative_difference) <= REFERENCE_RELATIVE_TOL
        if has_literature_reference
        else None
    )

    stopping_pass = converged

    case_pass = stopping_pass and compliance_pass is not False

    # =========================================================================
    # Final case report
    # =========================================================================

    print()
    print("=" * 100)
    print(f"FINAL RESULT — {case.topax_label}")
    print("=" * 100)

    print(f"paper mesh:            {case.paper_label}")

    print(f"TopaX mesh:            {case.topax_label}")

    print()

    print(f"converged:             {converged}")

    print(f"iterations:            {iteration}")

    print(f"final design change:   {final_change:.10e}")

    print(f"paper change criterion: {CHANGE_TOL:.10e}")

    if final_plateau_relative_change is not None:
        print(
            "final compliance drift: "
            f"{final_plateau_relative_change:.10e} "
            f"over {PLATEAU_WINDOW} iterations"
        )

    print(f"stopping reason:       {stopping_reason}")

    print(f"stopping criterion:    {'PASS' if stopping_pass else 'FAIL'}")

    print(f"setup time:            {setup_time:.4f} s")

    print(f"compilation time:      {compilation_time:.4f} s")

    print(f"initial evaluation:    {initial_evaluation_time:.4f} s")

    print(f"dummy execution:       {warmup_time:.4f} s")

    print(f"compile/warm-up time:  {compile_warmup_time:.4f} s")

    print(f"optimization time:     {optimization_time:.4f} s")

    print(f"optimization wall:     {optimization_wall_time:.4f} s")

    print(f"average iteration:     {optimization_time / iteration:.6f} s")

    print(f"final evaluation time: {final_evaluation_time:.4f} s")

    print()

    print(f"initial compliance C0: {objective_scale:.10f}")

    print(f"raw compliance C:      {raw_compliance:.10f}")

    print(f"normalized C/C0:       {normalized_compliance:.10f}")

    print()

    print(f"comparison t*C:        {comparison_compliance:.10f}")

    if has_literature_reference:
        print(f"reference compliance:  {case.reference_compliance:.10f}")

        print(f"absolute difference:   {absolute_difference:+.10f}")

        print(f"relative difference:   {relative_difference:+.6e}")

        print(f"relative error:        {relative_error_percent:+.6f}%")

        print(f"1% compliance check:   {'PASS' if compliance_pass else 'FAIL'}")
    else:
        print("reference compliance:  n/a (scaled exposition case)")

    print()

    print(f"mean design rho:       {float(final_mean_design_rho):.10f}")

    print(f"mean physical rho:     {float(final_mean_physical_rho):.10f}")

    print(f"normalized volume:     {float(final_normalized_volume):.10f}")

    print(f"physical-design mean:  {float(final_filter_volume_difference):+.6e}")

    print(f"constraint:            {float(final_constraint.value):+.6e}")

    print()

    print(f"Mnd diagnostic:        {float(final_mnd):.6f}%")

    print()

    print(
        f"design rho min/max:    {float(jnp.min(rho)):.6e} / {float(jnp.max(rho)):.6e}"
    )

    print(
        f"physical rho min/max:  "
        f"{float(jnp.min(final_rho_physical)):.6e} / "
        f"{float(jnp.max(final_rho_physical)):.6e}"
    )

    print()

    print(f"CASE RESULT:           {'PASS' if case_pass else 'FAIL'}")

    # =========================================================================
    # Save density fields
    # =========================================================================

    output_start = time.perf_counter()
    save_case_results(
        case,
        rho,
        final_rho_physical,
    )
    save_case_history(case, history, runtime_csv)
    output_time = time.perf_counter() - output_start
    topax.solver.close_solver_state(solvers)

    # =========================================================================
    # Return only scalars / metadata.
    #
    # Do not retain large JAX arrays across mesh cases.
    # =========================================================================

    case_wall_time = time.perf_counter() - case_wall_start
    coarse_factor = 2 ** (case.n_levels - 1)

    def iteration_statistics(key):
        values = [row[key] for row in history if row[key] != ""]
        return int(sum(values)), float(np.mean(values)), int(max(values))

    physics_cg = iteration_statistics("physics_cg_iterations")
    helmholtz_forward_cg = iteration_statistics(
        "helmholtz_forward_cg_iterations"
    )
    helmholtz_objective_cg = iteration_statistics(
        "helmholtz_objective_adjoint_cg_iterations"
    )
    helmholtz_constraint_cg = iteration_statistics(
        "helmholtz_constraint_adjoint_cg_iterations"
    )
    result = {
        "benchmark": "lazarov_2011",
        "case": case.topax_label,
        "literature_case": bool(has_literature_reference),
        "paper_mesh": case.paper_label,
        "topax_mesh": case.topax_label,
        "nx": case.nx,
        "ny": case.ny,
        "n_elements": (case.nx * case.ny),
        "n_displacement_dofs": int(2 * (case.nx + 1) * (case.ny + 1)),
        "n_levels": int(case.n_levels),
        "galerkin_crossover_level": int(case.n_levels - 1),
        "coarse_nx": int(case.nx // coarse_factor),
        "coarse_ny": int(case.ny // coarse_factor),
        "cg_tolerance": float(TOL),
        "cg_max_iterations": int(MAXITER),
        "optimization_method": "OC",
        "optimization_max_iterations": int(MAX_OPT_ITER),
        "change_tolerance": float(CHANGE_TOL),
        "compliance_plateau_min_iterations": int(PLATEAU_MIN_ITER),
        "compliance_plateau_window": int(PLATEAU_WINDOW),
        "compliance_plateau_relative_tolerance": float(PLATEAU_REL_TOL),
        "filter_type": "Helmholtz PDE density filter",
        "paper_filter_radius": float(PAPER_HELMHOLTZ_R),
        "topax_filter_radius": float(FILTER_RADIUS),
        "converged": bool(converged),
        "stopping_reason": stopping_reason,
        "iterations": int(iteration),
        "final_change": float(final_change),
        "final_compliance_plateau_relative_change": (
            final_plateau_relative_change
        ),
        "setup_time_s": float(setup_time),
        "compilation_time_s": float(compilation_time),
        "initial_evaluation_time_s": float(initial_evaluation_time),
        "warmup_time_s": float(warmup_time),
        "compile_warmup_time_s": float(compile_warmup_time),
        "optimization_time_s": float(optimization_time),
        "sum_iteration_time_s": float(optimization_time),
        "optimization_loop_wall_time_s": float(optimization_wall_time),
        "average_iteration_time_s": float(optimization_time / iteration),
        "final_evaluation_time_s": float(final_evaluation_time),
        "output_time_s": float(output_time),
        "case_wall_time_s": float(case_wall_time),
        "design_density_file": case_output_paths(case)["rho"].name,
        "physical_density_file": case_output_paths(case)["rho_physical"].name,
        "history_file": case_output_paths(case)["history"].name,
        "initial_compliance": float(objective_scale),
        # Standardized comparison quantity; retain the unscaled objective in
        # raw_compliance below.
        "final_compliance": float(comparison_compliance),
        "raw_compliance": float(raw_compliance),
        "normalized_compliance": float(normalized_compliance),
        "comparison_compliance": float(comparison_compliance),
        "reference_compliance": case.reference_compliance,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "relative_error_percent": relative_error_percent,
        "compliance_relative_error_percent": relative_error_percent,
        "mean_design_rho": float(final_mean_design_rho),
        "mean_physical_rho": float(final_mean_physical_rho),
        "normalized_volume": float(final_normalized_volume),
        "constraint": float(final_constraint.value),
        "mnd_percent": float(final_mnd),
        "stopping_pass": bool(stopping_pass),
        "compliance_pass": compliance_pass,
        "case_pass": bool(case_pass),
        "total_physics_cg_iterations": physics_cg[0],
        "average_physics_cg_iterations": physics_cg[1],
        "maximum_physics_cg_iterations": physics_cg[2],
        "total_helmholtz_forward_cg_iterations": helmholtz_forward_cg[0],
        "average_helmholtz_forward_cg_iterations": helmholtz_forward_cg[1],
        "maximum_helmholtz_forward_cg_iterations": helmholtz_forward_cg[2],
        "total_helmholtz_objective_adjoint_cg_iterations": (
            helmholtz_objective_cg[0]
        ),
        "average_helmholtz_objective_adjoint_cg_iterations": (
            helmholtz_objective_cg[1]
        ),
        "maximum_helmholtz_objective_adjoint_cg_iterations": (
            helmholtz_objective_cg[2]
        ),
        "total_helmholtz_constraint_adjoint_cg_iterations": (
            helmholtz_constraint_cg[0]
        ),
        "average_helmholtz_constraint_adjoint_cg_iterations": (
            helmholtz_constraint_cg[1]
        ),
        "maximum_helmholtz_constraint_adjoint_cg_iterations": (
            helmholtz_constraint_cg[2]
        ),
        "final_physics_cg_iterations": int(
            final_linear_solve_diagnostics.physics["n_iter"]
        ),
        "final_physics_cg_relative_residual": float(
            final_linear_solve_diagnostics.physics["relative_residual"]
        ),
    }
    for name, diagnostics in (
        ("helmholtz_forward", final_linear_solve_diagnostics.helmholtz_forward),
        (
            "helmholtz_objective_adjoint",
            final_linear_solve_diagnostics.helmholtz_objective_adjoint,
        ),
        (
            "helmholtz_constraint_adjoint",
            final_linear_solve_diagnostics.helmholtz_constraint_adjoint,
        ),
    ):
        result[f"final_{name}_cg_iterations"] = (
            int(diagnostics["n_iter"]) if diagnostics is not None else ""
        )
        result[f"final_{name}_cg_relative_residual"] = (
            float(diagnostics["relative_residual"])
            if diagnostics is not None
            else ""
        )
    result.update(runtime_metadata())
    return result


# =============================================================================
# Summary output
# =============================================================================


def print_summary(
    results,
):
    print()
    print()
    print("=" * 150)
    print("LAZAROV & SIGMUND (2011) — HELMHOLTZ DENSITY-FILTER SUMMARY")
    print("=" * 150)

    header = (
        f"{'paper mesh':>12} | "
        f"{'TopaX mesh':>12} | "
        f"{'elems':>9} | "
        f"{'iter':>5} | "
        f"{'conv':>5} | "
        f"{'C raw':>12} | "
        f"{'C/C0':>9} | "
        f"{'t*C':>10} | "
        f"{'C ref':>10} | "
        f"{'err %':>9} | "
        f"{'vol':>9} | "
        f"{'Mnd %':>9} | "
        f"{'result':>6}"
    )

    print(header)

    print("-" * len(header))

    for result in results:
        reference = result["reference_compliance"]
        relative_error = result["relative_error_percent"]
        reference_text = (
            f"{reference:10.6f}"
            if reference is not None
            else f"{'n/a':>10}"
        )
        error_text = (
            f"{relative_error:+9.4f}"
            if relative_error is not None
            else f"{'n/a':>9}"
        )
        print(
            f"{result['paper_mesh']:>12} | "
            f"{result['topax_mesh']:>12} | "
            f"{result['n_elements']:9d} | "
            f"{result['iterations']:5d} | "
            f"{str(result['converged']):>5} | "
            f"{result['raw_compliance']:12.4f} | "
            f"{result['normalized_compliance']:9.6f} | "
            f"{result['comparison_compliance']:10.6f} | "
            f"{reference_text} | "
            f"{error_text} | "
            f"{result['mean_physical_rho']:9.6f} | "
            f"{result['mnd_percent']:9.4f} | "
            f"{'PASS' if result['case_pass'] else 'FAIL':>6}"
        )

    print()
    print("-" * 150)

    computed = np.array(
        [result["comparison_compliance"] for result in results],
        dtype=np.float64,
    )

    reference_results = [
        result for result in results
        if result["reference_compliance"] is not None
    ]
    references = np.array(
        [result["reference_compliance"] for result in reference_results],
        dtype=np.float64,
    )

    absolute_relative_errors = np.abs(
        np.array(
            [result["relative_difference"] for result in reference_results],
            dtype=np.float64,
        )
    )

    computed_mean = float(np.mean(computed))

    computed_span = float(np.max(computed) - np.min(computed))

    computed_relative_span = computed_span / computed_mean

    reference_mean = float(np.mean(references)) if references.size else None
    reference_span = (
        float(np.max(references) - np.min(references))
        if references.size
        else None
    )
    reference_relative_span = (
        reference_span / reference_mean
        if references.size
        else None
    )
    max_abs_relative_error = (
        float(np.max(absolute_relative_errors))
        if absolute_relative_errors.size
        else None
    )

    all_converged = all(result["converged"] for result in results)

    all_stopping_pass = all(result["stopping_pass"] for result in results)

    all_compliance_pass = all(
        result["compliance_pass"] is not False for result in results
    )

    overall_pass = all_stopping_pass and all_compliance_pass

    print(
        f"computed t*C min/max:        "
        f"{np.min(computed):.10f} / "
        f"{np.max(computed):.10f}"
    )

    print(f"computed t*C span:           {computed_span:.10f}")

    print(f"computed relative span:      {100.0 * computed_relative_span:.6f}%")

    print()

    if references.size:
        print(
            f"paper C min/max:             "
            f"{np.min(references):.10f} / "
            f"{np.max(references):.10f}"
        )

        print(f"paper C span:                {reference_span:.10f}")

        print(
            f"paper relative span:         "
            f"{100.0 * reference_relative_span:.6f}%"
        )

        print()

        print(
            f"maximum |reference error|:   "
            f"{100.0 * max_abs_relative_error:.6f}%"
        )

    print(f"reference tolerance:         {100.0 * REFERENCE_RELATIVE_TOL:.6f}%")

    print()

    print(f"all cases converged:         {all_converged}")

    print(f"all stopping criteria pass:  {all_stopping_pass}")

    print(f"all compliance checks pass:  {all_compliance_pass}")

    print()

    print("=" * 150)

    print(f"OVERALL LAZAROV BENCHMARK:   {'PASS' if overall_pass else 'FAIL'}")

    print("=" * 150)

    return {
        "computed_compliance_mean": float(computed_mean),
        "computed_compliance_span": float(computed_span),
        "computed_relative_span": float(computed_relative_span),
        "reference_compliance_mean": reference_mean,
        "reference_compliance_span": reference_span,
        "reference_relative_span": reference_relative_span,
        "max_abs_relative_error": max_abs_relative_error,
        "all_converged": bool(all_converged),
        "all_stopping_pass": bool(all_stopping_pass),
        "all_compliance_pass": bool(all_compliance_pass),
        "overall_pass": bool(overall_pass),
    }


def save_summary(
    results,
    aggregate,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # CSV: one row per mesh.
    # -------------------------------------------------------------------------

    fieldnames = list(results[0].keys())

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

    # -------------------------------------------------------------------------
    # JSON: complete result + aggregate diagnostics.
    # -------------------------------------------------------------------------

    payload = {
        "paper": ("Lazarov & Sigmund (2011), Figure 8"),
        "paper_helmholtz_r": (PAPER_HELMHOLTZ_R),
        "topax_filter_radius": float(FILTER_RADIUS),
        "thickness": (THICKNESS),
        "volume_fraction": (VOLFRAC),
        "simp_penalty": (SIMP_PENALTY),
        "design_rho_min": (DESIGN_RHO_MIN),
        "topax_stiffness_floor": (RHO_MIN),
        "change_tolerance": (CHANGE_TOL),
        "reference_relative_tolerance": (REFERENCE_RELATIVE_TOL),
        "cases": (results),
        "aggregate": (aggregate),
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print()

    print(f"saved summary CSV:      {SUMMARY_CSV}")

    print(f"saved summary JSON:     {SUMMARY_JSON}")


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
    platform_name = "cuda" if args.device == "gpu" else "cpu"
    jax.config.update("jax_platform_name", platform_name)
    configure_output_dir(args.output_dir)

    selected = set(args.case or ())
    cases = tuple(
        case
        for case in CASES
        if (
            case.topax_label in selected
            or (not selected and case in LITERATURE_CASES)
        )
    )
    if args.n_levels is not None:
        cases = tuple(
            replace(case, n_levels=args.n_levels) for case in cases
        )
    if args.runtime_csv is not None and len(cases) != 1:
        raise ValueError("--runtime-csv requires exactly one selected case.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print("LAZAROV & SIGMUND (2011) — FIGURE 8 — HELMHOLTZ DENSITY FILTER")
    print("=" * 100)

    print()

    print("Mesh cases:")

    for case in cases:
        reference = (
            f"{case.reference_compliance:.6f}"
            if case.reference_compliance is not None
            else "n/a"
        )
        print(
            f"    paper {case.paper_label:>8} "
            f"<-> TopaX {case.topax_label:>8} "
            f"| levels = {case.n_levels} "
            f"| C_ref = {reference}"
        )

    print()

    print(f"r:                     {PAPER_HELMHOLTZ_R}")

    print(f"volume fraction:       {VOLFRAC}")

    print(f"change tolerance:      {CHANGE_TOL}")

    print(f"reference error tol:   {100.0 * REFERENCE_RELATIVE_TOL:.2f}%")

    print(f"device:                {jax.devices()[0]}")

    print(f"x64 enabled:           {jax.config.jax_enable_x64}")

    results = []

    complete_start = time.perf_counter()

    for case_index, case in enumerate(
        cases,
        start=1,
    ):
        print()
        print(f"Running case {case_index}/{len(cases)}...")

        result = run_case(case, runtime_csv=args.runtime_csv)

        results.append(result)

        # ---------------------------------------------------------------------
        # Avoid retaining compiled executables and Python references for all
        # three differently shaped problems simultaneously.
        #
        # Each mesh requires a separate JAX compilation anyway.
        # ---------------------------------------------------------------------

        gc.collect()

        jax.clear_caches()

    complete_time = time.perf_counter() - complete_start

    aggregate = print_summary(results)

    aggregate["total_wall_time_s"] = float(complete_time)

    print()

    print(f"total benchmark wall time: {complete_time:.4f} s")

    save_summary(
        results,
        aggregate,
    )


if __name__ == "__main__":
    main()
