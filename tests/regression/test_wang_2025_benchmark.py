import itertools
from pathlib import Path
import runpy

import jax.numpy as jnp
import numpy as np
import pytest

import topax


BENCHMARK = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "benchmark_scripts"
    / "topax"
    / "wang_2025"
    / "cantilever.py"
)


@pytest.fixture(scope="module")
def wang():
    return runpy.run_path(BENCHMARK)


def test_wang_resolution_and_filter_configuration(wang):
    resolve = wang["resolve_configuration"]

    standard = resolve(800)
    fixed = resolve(800, fixed_filter=True)

    np.testing.assert_array_equal(standard.elements, [800, 400, 400])
    assert standard.n_levels == 5
    assert standard.galerkin_crossover_level == 4
    assert standard.filter_radius_elements == pytest.approx(np.sqrt(3.0))
    assert standard.filter_radius == pytest.approx(np.sqrt(3.0) / 800.0)
    assert standard.helmholtz_radius == pytest.approx(np.sqrt(6.0) / 800.0)
    assert fixed.filter_radius_elements == pytest.approx(17.0 * np.sqrt(3.0))
    assert fixed.filter_radius == pytest.approx(17.0 * np.sqrt(3.0) / 800.0)
    assert fixed.helmholtz_radius == pytest.approx(
        17.0 * np.sqrt(6.0) / 800.0
    )


def test_wang_filter_matches_released_top3d_xl_element_matrix(wang):
    """The benchmark conversion reproduces TOP3D_XL's released filter."""

    configuration = wang["resolve_configuration"](48)
    grid = topax.construct_grid(2, wang["SIZES"], configuration.elements)
    helmholtz = topax.construct_helmholtz(
        filter_radius=configuration.helmholtz_radius,
        thickness=1.0,
    )
    actual = np.asarray(
        topax.construct_local_helmholtz(grid.fine_level, helmholtz)
    ).reshape(8, 8)

    g = 1.0 / np.sqrt(3.0)
    signs = np.asarray(tuple(itertools.product((-1.0, 1.0), repeat=3)))
    mass = np.zeros((8, 8))
    gradient = np.zeros((8, 8))
    for xi, eta, zeta in itertools.product((-g, g), repeat=3):
        sx, sy, sz = signs.T
        shape = 0.125 * (1 + sx * xi) * (1 + sy * eta) * (1 + sz * zeta)
        derivatives = np.stack(
            (
                0.125 * sx * (1 + sy * eta) * (1 + sz * zeta),
                0.125 * sy * (1 + sx * xi) * (1 + sz * zeta),
                0.125 * sz * (1 + sx * xi) * (1 + sy * eta),
            )
        )
        mass += np.outer(shape, shape) / 8.0
        gradient += derivatives.T @ derivatives

    top3d_xl = mass + 0.25 * gradient
    element_volume = float(np.prod(grid.fine_level.element_sizes))
    np.testing.assert_allclose(
        actual / element_volume,
        top3d_xl,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_wang_rejects_non_dyadic_r300(wang):
    with pytest.raises(ValueError, match="R300 is intentionally unsupported"):
        wang["resolve_configuration"](300)


def test_wang_rejects_incompatible_requested_hierarchy(wang):
    with pytest.raises(ValueError, match="not divisible"):
        wang["resolve_configuration"](800, n_levels=6)


def test_wang_r1024_supports_final_gpu_hierarchies(wang):
    for levels, crossover in ((4, 3), (5, 4)):
        configuration = wang["resolve_configuration"](
            1024,
            n_levels=levels,
            galerkin_crossover_level=crossover,
        )
        np.testing.assert_array_equal(configuration.elements, [1024, 512, 512])
        assert configuration.n_levels == levels
        assert configuration.galerkin_crossover_level == crossover


def test_wang_support_and_normalized_load_match_reference_geometry(wang):
    elements = np.array([12, 6, 6])
    grid = topax.construct_grid(2, wang["SIZES"], elements)
    elastic = topax.construct_elastic(1.0, 0.3)
    restrictions = topax.construct_restrictions(
        grid,
        elastic,
        wang["DIRICHLET_BOUNDARY_CONDITIONS"],
    )
    forces, n_loaded_nodes = wang["construct_wang_forces"](grid)

    assert not bool(jnp.any(restrictions[0][:, 0, :, :]))
    assert bool(jnp.all(restrictions[0][:, 1:, :, :]))
    assert n_loaded_nodes == 14
    assert float(jnp.sum(forces)) == pytest.approx(-1.0)
    assert not bool(jnp.any(forces[:2]))
    assert not bool(jnp.any(forces[2, :-1]))
    assert int(jnp.count_nonzero(forces[2, -1])) == n_loaded_nodes


def test_wang_tiny_problem_runs_complete_sensitivity_filtered_step(wang):
    original = np.array(wang["RESOLUTION_ELEMENTS"][48], copy=True)
    wang["RESOLUTION_ELEMENTS"][48] = np.array([8, 4, 4])
    try:
        result = wang["optimize"](
            resolution=48,
            n_levels=3,
            max_iterations=1,
            galerkin_crossover_level=2,
            galerkin_slab_size=1,
        )
    finally:
        wang["RESOLUTION_ELEMENTS"][48] = original

    assert result["iterations"] == 1
    assert np.isfinite(result["compliance"])
    assert result["compliance"] > 0.0
    assert np.isfinite(result["volume_constraint"])
    assert np.isfinite(result["mdn"])
    assert np.asarray(result["rho"]).shape == (8, 4, 4)
