# tests/conftest.py
from pathlib import Path
import pytest
import numpy as np

import afv


@pytest.fixture(scope="session")
def data_dir() -> Path:
    # Return the path to the data directory
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def phys() -> afv.PhysicalParams:
    return afv.PhysicalParams(
        r=1.0,
        A0=np.pi,
        P0=4.8,
        KA=1.0,
        KP=1.0,
        lambda_tension=0.2,
    )


@pytest.fixture(scope="session")
def simulator(phys) -> afv.FiniteVoronoiSimulator:
    # Initialize the Voronoi simulator with the defined parameters
    pts = np.array([[0.0, 0.0]])
    sim = afv.FiniteVoronoiSimulator(pts, phys)
    
    _USING_ACCEL = sim._BACKEND in {"cython", "numba"}

    assert _USING_ACCEL, "Accelerated backend is not in use."
    return sim
