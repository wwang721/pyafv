"""
PyAFV - A Python implementation of the active-finite-Voronoi (AFV) model in 2D.

**Classes**

.. autosummary::
   :nosignatures:

   PhysicalParams
   FiniteVoronoiSimulator

**Functions**

.. autosummary::
   :nosignatures:

   target_delta
"""

from .physical_params import PhysicalParams, target_delta
from .finite_voronoi import FiniteVoronoiSimulator

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


__all__ = [
    "PhysicalParams",
    "FiniteVoronoiSimulator",
    "target_delta",
]
