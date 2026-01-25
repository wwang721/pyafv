"""
This is a benchmark module for the numpy.asarray function.
"""

import numpy as np
from scipy.spatial import Voronoi
import pytest


@pytest.fixture(scope="module")
def ridge_vertices_list() -> list:
    rng = np.random.default_rng(42)
    pts = rng.random((1000, 2)) * np.sqrt(1000)
    vor = Voronoi(pts)
    return vor.ridge_vertices


@pytest.fixture(scope="module", params=["accel", "fallback"])
def list_to_Nx2_int_array(request):
    if request.param == "accel":
        from pyafv.backend import backend_impl, _BACKEND_NAME
        _USING_ACCEL = _BACKEND_NAME in {"cython", "numba"}
        assert _USING_ACCEL, "Accelerated backend is not in use."
        return backend_impl.list_to_Nx2_int_array
    else:
        from pyafv.cell_geom_fallback import list_to_Nx2_int_array
        return list_to_Nx2_int_array


def test_asarray(benchmark, list_to_Nx2_int_array, ridge_vertices_list):
    benchmark(list_to_Nx2_int_array, ridge_vertices_list)
