"""
Benchmarks for pyafv.ParallelFiniteVoronoiSimulator.build.

These benchmarks measure the domain-task construction step and the repeated
``build(connect=False)`` call. The worker pool is kept alive during the build
benchmark so that process-pool startup is not included.
"""

import numpy as np
import pytest

import pyafv as afv


@pytest.fixture(scope="module")
def parallel_pts() -> np.ndarray:
    rng = np.random.default_rng(42)
    n_points = 100_000
    return rng.random((n_points, 2)) * np.sqrt(n_points)


@pytest.fixture(scope="module")
def parallel_phys() -> afv.PhysicalParams:
    return afv.PhysicalParams(r=1.0)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((2, 2), id="2x2"),
        pytest.param((3, 3), id="3x3"),
        pytest.param((4, 3), id="4x3"),
    ],
)
def parallel_simulator(request, parallel_pts, parallel_phys):
    grid_shape = request.param
    n_workers = grid_shape[0] * grid_shape[1]
    sim = afv.ParallelFiniteVoronoiSimulator(
        parallel_pts,
        parallel_phys,
        grid_shape=grid_shape,
        n_workers=n_workers,
    )
    with sim:
        yield sim


def test_parallel_make_domain_tasks(benchmark, parallel_simulator):
    benchmark(
        parallel_simulator._make_domain_tasks,
        connect=False,
        plot_mode=False,
    )


def test_parallel_full_build(benchmark, parallel_simulator):
    benchmark(parallel_simulator.build, connect=False)
