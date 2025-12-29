import warnings


try:
    from . import finite_voronoi_fast as _impl
    _USING_ACCEL = True
except ImportError:             # pragma: no cover
    _USING_ACCEL = False
    # raise warning to inform user about fallback
    warnings.warn(
        "Could not import the accelerated fast Cython module. "
        "Falling back to the pure Python implementation, which may be slower. "
        "To enable the accelerated version, ensure that all dependencies are installed.",
        RuntimeWarning,
        stacklevel=2,
    )
    from . import finite_voronoi_fallback as _impl

# ---- explicit public API ----
PhysicalParams = _impl.PhysicalParams
FiniteVoronoiSimulator = _impl.FiniteVoronoiSimulator

__all__ = [
    "PhysicalParams",
    "FiniteVoronoiSimulator",
    "_USING_ACCEL",
]
