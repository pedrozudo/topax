import numpy as np
import pytest

from topax.grid import Grid, Level, construct_grid


def test_construct_grid_accepts_2d_and_uses_ceiling_coarsening():
    grid = construct_grid(
        n_levels=3,
        sizes=[2.0, 1.0],
        elements=[5, 4],
    )

    assert grid.n_dims == 2
    assert grid.n_levels == 3

    np.testing.assert_array_equal(grid.levels[0].elements, [5, 4])
    np.testing.assert_array_equal(grid.levels[1].elements, [3, 2])
    np.testing.assert_array_equal(grid.levels[2].elements, [2, 1])


def test_grid_accepts_3d():
    grid = Grid(
        n_levels=np.int64(2),
        sizes=[3.0, 2.0, 1.0],
        elements=[6.0, 4.0, 2.0],
    )

    assert grid.n_dims == 3
    assert grid.fine_level.n_nodes_element == 8
    np.testing.assert_array_equal(grid.coarse_level.elements, [3, 2, 1])
    np.testing.assert_allclose(grid.fine_level.element_sizes, [0.5, 0.5, 0.5])


def test_grid_preserves_size_dtype():
    grid = Grid(
        n_levels=1,
        sizes=np.array([1.0, 2.0], dtype=np.float32),
        elements=[2, 4],
    )

    assert grid.sizes.dtype == np.float32


@pytest.mark.parametrize("n_levels", [0, -1, 1.5, True])
def test_grid_rejects_invalid_level_counts(n_levels):
    with pytest.raises(ValueError, match="n_levels"):
        Grid(
            n_levels=n_levels,
            sizes=[1.0, 1.0],
            elements=[4, 4],
        )


@pytest.mark.parametrize(
    "sizes,elements,message",
    [
        ([1.0], [4], "Only 2D and 3D"),
        ([1.0, 1.0, 1.0, 1.0], [4, 4, 4, 4], "Only 2D and 3D"),
        ([[1.0, 1.0]], [4, 4], "sizes must be one-dimensional"),
        ([1.0, 1.0], [[4, 4]], "elements must be one-dimensional"),
        ([1.0, 1.0, 1.0], [4, 4], "must have the same shape"),
    ],
)
def test_grid_rejects_invalid_geometry_shapes(sizes, elements, message):
    with pytest.raises(ValueError, match=message):
        Grid(
            n_levels=1,
            sizes=sizes,
            elements=elements,
        )


@pytest.mark.parametrize(
    "sizes,message",
    [
        ([0.0, 1.0], "strictly positive"),
        ([-1.0, 1.0], "strictly positive"),
        ([np.nan, 1.0], "finite"),
        ([np.inf, 1.0], "finite"),
        (["invalid", 1.0], "numeric"),
    ],
)
def test_grid_rejects_invalid_sizes(sizes, message):
    with pytest.raises(ValueError, match=message):
        Grid(
            n_levels=1,
            sizes=sizes,
            elements=[4, 4],
        )


@pytest.mark.parametrize(
    "elements,message",
    [
        ([0, 4], "strictly positive"),
        ([-1, 4], "strictly positive"),
        ([2.5, 4], "integer values"),
        ([np.nan, 4], "finite"),
        ([np.inf, 4], "finite"),
        ([True, False], "not booleans"),
        (["invalid", 4], "numeric"),
    ],
)
def test_grid_rejects_invalid_elements(elements, message):
    with pytest.raises(ValueError, match=message):
        Grid(
            n_levels=1,
            sizes=[1.0, 1.0],
            elements=elements,
        )


def test_level_applies_the_same_geometry_validation():
    with pytest.raises(ValueError, match="must have the same shape"):
        Level(
            sizes=[1.0, 1.0, 1.0],
            elements=[4, 4],
        )

@pytest.mark.parametrize(
    "sizes,elements,n_levels,max_levels",
    [
        ([1.0, 1.0], [1, 1], 2, 1),
        ([2.0, 1.0], [5, 4], 5, 4),
        ([3.0, 2.0, 1.0], [2, 3, 1], 4, 3),
    ],
)
def test_grid_rejects_hierarchies_deeper_than_one_element_per_axis(
    sizes, elements, n_levels, max_levels
):
    with pytest.raises(
        ValueError,
        match=rf"maximum useful hierarchy depth of {max_levels}",
    ):
        Grid(n_levels=n_levels, sizes=sizes, elements=elements)


@pytest.mark.parametrize(
    "sizes,elements,n_levels",
    [
        ([1.0, 1.0], [1, 1], 1),
        ([2.0, 1.0], [5, 4], 4),
        ([3.0, 2.0, 1.0], [2, 3, 1], 3),
    ],
)
def test_grid_accepts_maximum_useful_hierarchy_depth(
    sizes, elements, n_levels
):
    grid = Grid(n_levels=n_levels, sizes=sizes, elements=elements)

    np.testing.assert_array_equal(
        grid.coarse_level.elements,
        np.ones(len(elements), dtype=int),
    )
