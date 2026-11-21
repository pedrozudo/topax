import numpy as np


def _validate_sizes_and_elements(sizes, elements):
    try:
        sizes = np.asarray(sizes)
    except (TypeError, ValueError) as exc:
        raise ValueError("sizes must contain numeric values.") from exc

    if sizes.ndim != 1:
        raise ValueError(f"sizes must be one-dimensional, got shape {sizes.shape}.")

    if sizes.size not in (2, 3):
        raise ValueError(
            f"Only 2D and 3D grids are supported, got {sizes.size} dimensions."
        )

    if not np.issubdtype(sizes.dtype, np.number) or np.issubdtype(
        sizes.dtype, np.complexfloating
    ):
        raise ValueError("sizes must contain real numeric values.")

    if not np.all(np.isfinite(sizes)):
        raise ValueError("sizes must contain only finite values.")

    if np.any(sizes <= 0):
        raise ValueError("sizes must be strictly positive.")

    try:
        elements = np.asarray(elements)
    except (TypeError, ValueError) as exc:
        raise ValueError("elements must contain numeric values.") from exc

    if elements.dtype == np.bool_:
        raise ValueError("elements must contain positive integers, not booleans.")

    if elements.ndim != 1:
        raise ValueError(
            f"elements must be one-dimensional, got shape {elements.shape}."
        )

    if elements.shape != sizes.shape:
        raise ValueError(
            "sizes and elements must have the same shape, "
            f"got {sizes.shape} and {elements.shape}."
        )

    if not np.issubdtype(elements.dtype, np.number) or np.issubdtype(
        elements.dtype, np.complexfloating
    ):
        raise ValueError("elements must contain real numeric values.")

    if not np.all(np.isfinite(elements)):
        raise ValueError("elements must contain only finite values.")

    if np.any(elements <= 0):
        raise ValueError("elements must be strictly positive.")

    if np.any(elements != np.floor(elements)):
        raise ValueError("elements must contain integer values.")

    return sizes, elements.astype(int)


def construct_grid(*args, **kwargs):
    return Grid(*args, **kwargs)


class Grid:
    def __init__(self, n_levels, sizes: np.ndarray, elements: np.ndarray):
        if isinstance(n_levels, (bool, np.bool_)) or not isinstance(
            n_levels, (int, np.integer)
        ):
            raise ValueError("n_levels must be a positive integer.")

        if n_levels < 1:
            raise ValueError("n_levels must be at least 1.")

        self.n_levels = int(n_levels)
        self.sizes, self.elements = _validate_sizes_and_elements(sizes, elements)

        max_levels = 1
        coarsest_elements = self.elements
        while np.any(coarsest_elements > 1):
            coarsest_elements = coarsest_elements // 2 + coarsest_elements % 2
            max_levels += 1

        if self.n_levels > max_levels:
            raise ValueError(
                f"n_levels={self.n_levels} exceeds the maximum useful hierarchy "
                f"depth of {max_levels} for elements {self.elements.tolist()}."
            )

        self.levels = self.init_levels()

    def init_levels(self):
        elements = self.elements

        levels = []

        for _ in range(self.n_levels):
            levels.append(Level(self.sizes, elements))

            elements = elements // 2 + elements % 2

        return levels

    @property
    def n_dims(self):
        return self.sizes.shape[0]

    @property
    def fine_level(self):
        return self.levels[0]

    @property
    def coarse_level(self):
        return self.levels[-1]


class Level:
    def __init__(self, sizes: np.ndarray, elements: np.ndarray):
        self.sizes, self.elements = _validate_sizes_and_elements(sizes, elements)


    @property
    def n_dims(self):
        return self.elements.shape[0]

    @property
    def element_sizes(self):
        return self.sizes / self.elements

    @property
    def n_elements(self):
        return int(np.prod(self.elements))

    @property
    def nodes(self):
        return self.elements + 1

    @property
    def n_nodes(self):
        return int(np.prod(self.nodes))

    @property
    def n_nodes_element(self):
        return 2**self.n_dims
