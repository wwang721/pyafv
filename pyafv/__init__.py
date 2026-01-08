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
from .simulator import FiniteVoronoiSimulator

__version__ = "0.3.3"

__all__ = [
    "PhysicalParams",
    "FiniteVoronoiSimulator",
    "target_delta",
]
