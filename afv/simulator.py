# API-facing wrapper

from .backend import backend_simulator, _BACKEND_NAME


class FiniteVoronoiSimulator(backend_simulator):
    """Finite Voronoi Simulator.

    This class provides an interface to simulate finite Voronoi models.
    It wraps around the backend simulator implementation, which may be
    either a Cython-accelerated version or a pure Python fallback.

    Attributes:
        See `backend_simulator` for details.
    """
    # Define as a class attribute
    _BACKEND = _BACKEND_NAME

    def __init__(self, *args, **kwargs):
        # Ensure you call the parent constructor
        super().__init__(*args, **kwargs)

        if self._BACKEND not in {"cython", "numba"}:                 # pragma: no cover
            # raise warning to inform user about fallback
            import warnings
            warnings.warn(
                "Could not import the Cython-accelerated module. "
                "Falling back to the pure Python implementation, which may be slower. "
                "To enable the accelerated version, ensure that all dependencies are installed.",
                RuntimeWarning,
                stacklevel=2,
            )


__all__ = [
    "FiniteVoronoiSimulator",
]
