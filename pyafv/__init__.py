from .physical_params import PhysicalParams, target_delta
from .simulator import FiniteVoronoiSimulator

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyafv")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "PhysicalParams",
    "FiniteVoronoiSimulator",
    "target_delta",
]
