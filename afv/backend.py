# chooses fast vs fallback implementation

try:
    from . import finite_voronoi_fast as _impl
    _BACKEND_NAME = "cython"
except ImportError:             # pragma: no cover
    _BACKEND_NAME = "python"
    from . import finite_voronoi_fallback as _impl

# ---- for explicit API ----
backend_simulator = _impl.FiniteVoronoiSimulator

__all__ = [
    "backend_simulator",
    "_BACKEND_NAME",
]
