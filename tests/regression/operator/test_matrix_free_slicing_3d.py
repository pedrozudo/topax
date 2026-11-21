import jax
import jax.numpy as jnp
import numpy as np
import pytest

from topax.boundary import construct_restrictions
from topax.grid import Grid
from topax.material.elastic import construct_elastic, construct_local_stiffness
from topax.material.helmholtz import construct_helmholtz, construct_local_helmholtz
from topax.operator import matrix_free_matvec, physics_weighing
from topax import vcycle as vcycle_module


jax.config.update("jax_enable_x64", True)


def construct_physics(elements, dtype):
    elements = np.asarray(elements)
    grid = Grid(
        n_levels=1,
        sizes=np.asarray([2.3, 1.1, 0.7], dtype=dtype),
        elements=elements,
    )
    elastic = construct_elastic(1.7, 0.27, 1.0)
    local = construct_local_stiffness(grid.fine_level, elastic).astype(dtype)
    fixed_face = {"center": [0.0, 0.5, 0.5], "size": [0.0, 1.0, 1.0]}
    restriction = construct_restrictions(
        grid,
        elastic,
        [[fixed_face], [fixed_face], [fixed_face]],
    )[0]
    rho = jnp.linspace(
        0.25,
        0.9,
        grid.fine_level.n_elements,
        dtype=dtype,
    ).reshape(tuple(elements))
    x = jnp.sin(
        jnp.arange(3 * grid.fine_level.n_nodes, dtype=dtype) / 7.0
    ).reshape((3, *grid.fine_level.nodes))
    return grid.fine_level, local, restriction, rho, x


@pytest.mark.parametrize(
    ("elements", "slab_size", "slab_axis"),
    [
        ((1, 1, 1), 2, None),
        ((2, 1, 2), 2, 0),
        ((2, 4, 3), 2, 1),
        ((2, 3, 5), 3, 2),
        ((4, 3, 2), 2, None),
    ],
)
@pytest.mark.parametrize("with_restrictions", [False, True])
def test_3d_slab_configurations_match_single_slab_under_jit(
    elements,
    slab_size,
    slab_axis,
    with_restrictions,
):
    level, local, restriction, rho, x = construct_physics(
        elements,
        jnp.float64,
    )
    weighing = physics_weighing(rho, 3.0, 1e-6)
    restriction = restriction if with_restrictions else None

    @jax.jit
    def apply(x, weighing):
        return matrix_free_matvec(
            level,
            x,
            local,
            weighing,
            restriction,
            slab_size=slab_size,
            slab_axis=slab_axis,
        )

    actual = apply(x, weighing)
    expected = matrix_free_matvec(
        level,
        x,
        local,
        weighing,
        restriction,
        slab_size=max(map(int, level.nodes)),
        slab_axis=slab_axis,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_3d_helmholtz_multiple_slabs_match_single_slab():
    elements = np.array([2, 3, 5])
    grid = Grid(1, np.array([1.3, 0.8, 1.7]), elements)
    helmholtz = construct_helmholtz(filter_radius=0.6, thickness=4.0)
    local = construct_local_helmholtz(grid.fine_level, helmholtz)
    rho_mask = jnp.linspace(0.3, 1.0, np.prod(elements)).reshape(tuple(elements))
    x = jnp.cos(jnp.arange(grid.fine_level.n_nodes) / 5.0).reshape(
        (1, *grid.fine_level.nodes)
    )

    actual = jax.jit(
        lambda x, rho_mask: matrix_free_matvec(
            grid.fine_level,
            x,
            local,
            rho_mask,
            slab_size=2,
            slab_axis=2,
        )
    )(x, rho_mask)
    expected = matrix_free_matvec(
        grid.fine_level,
        x,
        local,
        rho_mask,
        slab_size=max(map(int, grid.fine_level.nodes)),
        slab_axis=2,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_3d_multiple_slab_vjp_matches_single_slab():
    level, local, restriction, rho, x = construct_physics(
        (2, 1, 2),
        jnp.float64,
    )
    cotangent = jnp.cos(jnp.arange(x.size) / 11.0).reshape(x.shape)

    def scalar_output(x, rho, **options):
        return jnp.vdot(
            cotangent,
            matrix_free_matvec(
                level,
                x,
                local,
                physics_weighing(rho, 3.0, 1e-6),
                restriction,
                **options,
            ),
        )

    expected = jax.grad(
        lambda x, rho: scalar_output(
            x,
            rho,
            slab_size=max(map(int, level.nodes)),
            slab_axis=2,
        ),
        argnums=(0, 1),
    )(x, rho)
    actual = jax.jit(
        jax.grad(
            lambda x, rho: scalar_output(
                x,
                rho,
                slab_size=2,
                slab_axis=2,
            ),
            argnums=(0, 1),
        )
    )(x, rho)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-11, atol=1e-11)


def test_physics_solve_propagates_fine_slab_options(monkeypatch):
    grid = Grid(
        n_levels=2,
        sizes=np.ones(3),
        elements=np.array([2, 2, 2]),
    )
    nodal = jnp.zeros((3, *grid.fine_level.nodes), dtype=jnp.float64)
    restrictions = tuple(jnp.ones_like(nodal, dtype=jnp.bool_) for _ in grid.levels)
    calls = []

    def fake_construct_multigrid_data(*args, **kwargs):
        del args, kwargs
        return None, (None,), (None,), (jnp.ones_like(nodal),)

    def fake_matrix_free_matvec(
        level,
        value,
        kernel,
        weighing,
        restriction=None,
        *,
        slab_size,
        slab_axis,
        rematerialize=True,
    ):
        del level, kernel, weighing, restriction, rematerialize
        calls.append((slab_size, slab_axis))
        return value

    def fake_cg_solve_stateful(
        solver,
        matvec,
        right_hand_side,
        x0,
        *,
        precondition,
        **kwargs,
    ):
        del right_hand_side, kwargs
        matvec(x0)
        fine_preconditioner_matvec = precondition.keywords[
            "level_matvec_impls"
        ][0]
        fine_preconditioner_matvec(
            grid.fine_level,
            x0,
            None,
            None,
            restrictions[0],
        )
        return solver, x0, {
            "error": jnp.asarray(0.0),
            "converged": jnp.asarray(True),
            "n_iter": jnp.asarray(0),
        }

    monkeypatch.setattr(
        vcycle_module,
        "construct_physics_multigrid_data",
        fake_construct_multigrid_data,
    )
    monkeypatch.setattr(
        vcycle_module,
        "matrix_free_matvec",
        fake_matrix_free_matvec,
    )
    monkeypatch.setattr(
        vcycle_module,
        "cg_solve_stateful",
        fake_cg_solve_stateful,
    )

    vcycle_module.physics_vcycle_solve(
        solver=None,
        grid=grid,
        fine_local_operator=None,
        coarse_structure=None,
        rho=jnp.ones(tuple(grid.fine_level.elements)),
        forces=nodal,
        restrictions=restrictions,
        x0=nodal,
        simp_penalty=3.0,
        rho_min=1e-6,
        damping=0.4,
        tol=1e-5,
        maxiter=10,
        slab_size=24,
        slab_axis=2,
    )

    assert calls == [(24, 2), (24, 2)]
