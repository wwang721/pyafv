"""
This module contains benchmarks for the pyafv.FiniteVolumeSimulator.build method.
The build method contains two main routines:
    * _build_voronoi_with_extensions
    * _per_cell_geometry
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def initial_pts() -> np.ndarray:
    rng = np.random.default_rng(42)
    pts = rng.random((1000, 2)) * np.sqrt(1000)
    return pts

@pytest.fixture(scope="module")
def result_build_Voronoi(simulator, initial_pts):
    simulator.update_positions(initial_pts)
    (vor, vertices_all, ridge_vertices_all, num_vertices,
            vertexpair2ridge, vertex_points) = simulator._build_voronoi_with_extensions()
    return (vor, vertices_all, ridge_vertices_all, num_vertices,
            vertexpair2ridge, vertex_points)


def test_build_voronoi(benchmark, simulator, initial_pts):

    simulator.update_positions(initial_pts)
    benchmark(simulator._build_voronoi_with_extensions)

def test_cell_geometry(benchmark, simulator, result_build_Voronoi):
    (vor, vertices_all, ridge_vertices_all,
     num_vertices, vertexpair2ridge, vertex_points) = result_build_Voronoi

    benchmark(
        simulator._per_cell_geometry,
        vor,
        vertices_all,
        ridge_vertices_all,
        num_vertices,
        vertexpair2ridge
    )

def test_full_build(benchmark, simulator, initial_pts):
    simulator.update_positions(initial_pts)
    benchmark(simulator.build)
