"""
PyAFV - A Python implementation of the **active-finite-Voronoi (AFV) model** in 2D.
"""

from .physical_params import PhysicalParams, target_delta
from .finite_voronoi import FiniteVoronoiSimulator
from . import calibrate

try:
    from ._version import __version__
except ImportError:                          # pragma: no cover
    __version__ = "unknown"


__all__ = [
    "PhysicalParams",
    "FiniteVoronoiSimulator",
    "target_delta",
    "calibrate",
]
