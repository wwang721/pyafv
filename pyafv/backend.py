# chooses fast vs fallback implementation

try:
    from . import cell_geom as _impl
    _BACKEND_NAME = "cython"
    _IMPORT_ERROR = None
except Exception as e:             # pragma: no cover
    from . import cell_geom_fallback as _impl
    _BACKEND_NAME = "python"
    _IMPORT_ERROR = e


# ---- for explicit API ----
backend_impl = _impl

__all__ = [
    "backend_impl",
    "_BACKEND_NAME",
    "_IMPORT_ERROR"
]
