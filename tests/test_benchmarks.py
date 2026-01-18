"""
This module contains benchmarks for the pyafv.FiniteVolumeSimulator.build method.
The build method contains three main routines:
    * _build_voronoi_with_extensions
    * _per_cell_geometry
    * _assemble_forces
    * _get_connections
The first two routines depend on specific backends.
"""

import numpy as np
import pytest
import pyafv as afv


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

def test_get_connections(benchmark, initial_pts):

    sim = afv.FiniteVoronoiSimulator(initial_pts, afv.PhysicalParams())
    (vor, vertices_all, ridge_vertices_all, num_vertices,
        vertexpair2ridge, vertex_points) = sim._build_voronoi_with_extensions()
    benchmark(sim._get_connections, vor.ridge_points, vertices_all, ridge_vertices_all)

def test_assemble_forces(benchmark, initial_pts):
    
    sim = afv.FiniteVoronoiSimulator(initial_pts, afv.PhysicalParams())
    (vor, vertices_all, ridge_vertices_all, num_vertices,
        vertexpair2ridge, vertex_points) = sim._build_voronoi_with_extensions()
    geom, vertices_all = sim._per_cell_geometry(
        vor, vertices_all, ridge_vertices_all, num_vertices, vertexpair2ridge)

    benchmark(sim._assemble_forces,
        vertices_all=vertices_all,
        num_vertices_ext=geom["num_vertices_ext"],
        vertex_points=vertex_points,
        vertex_in_id=list(geom["vertex_in_id"]),
        vertex_out_id=list(geom["vertex_out_id"]),
        vertex_out_points=geom["vertex_out_points"],
        vertex_out_da_dtheta=geom["vertex_out_da_dtheta"],
        vertex_out_dl_dtheta=geom["vertex_out_dl_dtheta"],
        dA_poly_dh=geom["dA_poly_dh"],
        dP_poly_dh=geom["dP_poly_dh"],
        area_list=geom["area_list"],
        perimeter_list=geom["perimeter_list"],
    )

def test_full_build(benchmark, simulator, initial_pts):
    simulator.update_positions(initial_pts)
    benchmark(simulator.build)
