"""
Benchmarks for pyafv.decompose_points halo-collection methods.

The parameter grid compares the available decomposition methods over several
system sizes and domain grids. Benchmarks use fixed point sets so methods see
the same input for a given system size.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyafv as afv


SYSTEM_SIZES = [10_000, 100_000, 1_000_000]
GRID_SHAPES = [(4, 3), (8, 6), (16, 12)]
METHODS = ["dense", "binned", "sorted_x"]
RADIUS = 1.0
HALO_WIDTH = 4.01 * RADIUS
SEED = 42


@pytest.fixture(scope="module")
def point_sets() -> dict[int, np.ndarray]:
    return {
        n_points: make_points(n_points)
        for n_points in SYSTEM_SIZES
    }


def make_points(n_points: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + n_points)
    box_size = np.sqrt(n_points) * RADIUS
    return rng.random((n_points, 2)) * box_size


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("grid_shape", GRID_SHAPES, ids=lambda grid: f"{grid[0]}x{grid[1]}")
@pytest.mark.parametrize("n_points", SYSTEM_SIZES, ids=lambda n: f"N={n}")
def test_decompose_points_methods(benchmark, point_sets, n_points, grid_shape, method):
    points = point_sets[n_points]
    benchmark.pedantic(
        afv.decompose_points,
        args=(points, grid_shape, HALO_WIDTH),
        kwargs={"method": method},
        rounds=5,
        iterations=1,
    )
