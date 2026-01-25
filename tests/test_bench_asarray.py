"""
This is a benchmark module for the numpy.asarray function.
"""

import numpy as np
import pytest
from pyafv.cell_geom import list_to_Nx2_int_array

@pytest.fixture(scope="module")
def initial_pts() -> np.ndarray:
    rng = np.random.default_rng(42)
    pts = rng.random((1000, 2)) * np.sqrt(1000)
    return pts

@pytest.fixture(scope="module")
def ridge_vertices_list(initial_pts) -> list:
    from scipy.spatial import Voronoi
    vor = Voronoi(initial_pts)
    return vor.ridge_vertices

def test_asarray(benchmark, ridge_vertices_list):
    benchmark(np.asarray, ridge_vertices_list, dtype=int)

def test_cython_conversion(benchmark, ridge_vertices_list):
    benchmark(list_to_Nx2_int_array, ridge_vertices_list)
