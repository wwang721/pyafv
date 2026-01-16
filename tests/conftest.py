# tests/conftest.py
from pathlib import Path
import pytest
import numpy as np

import pyafv as afv


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
        delta=None,
    )


@pytest.fixture(scope="session", params=["accel", "fallback"])
def simulator(request, phys):
    """Fixture that provides both the accelerated and fallback simulators."""
    pts = np.array([[0.0, 0.0]])

    if request.param == "accel":
        sim = afv.FiniteVoronoiSimulator(pts, phys)
        # Verify it's actually using an accelerated backend
        _USING_ACCEL = sim._BACKEND in {"cython", "numba"}

        assert _USING_ACCEL, "Accelerated backend is not in use."
        return sim

    else:
        sim = afv.FiniteVoronoiSimulator(pts, phys, backend="python")

        assert sim._BACKEND == "python", "Fallback backend is not in use."
        return sim
