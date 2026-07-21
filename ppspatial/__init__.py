"""ppspatial — postpyc reimplementation of scipy.spatial.

Each function is a POST Python kernel written in fully-typed postpyc. The
compiler lowers the kernels to native shared-library code; in interpreted
mode they run via the pure-Python broadcast loop. When the optional
``ppspatial_native`` extension module is installed next to this package,
matching public functions are replaced with native NumPy ufuncs at import
time.

Function families implemented
------------------------------
Distance (_distance) : euclidean, sqeuclidean, cityblock, chebyshev,
                       cosine, correlation, minkowski

Roadmap (not yet implemented)
------------------------------
Distance             : cdist
Transforms           : Rotation core ops on Shape[4] quaternions
Blocked on compiler  : pdist condensed form (computed core dim n(n-1)/2),
                       KDTree / ConvexHull / Delaunay
See ROADMAP.md for status and upstream postpython requests.
"""

from importlib import import_module as _import_module
from warnings import warn as _warn

from ppspatial._distance import (
    euclidean,
    sqeuclidean,
    cityblock,
    chebyshev,
    cosine,
    correlation,
    minkowski,
)

__all__ = [
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
    "cosine",
    "correlation",
    "minkowski",
]

# __native_available__ is True when AT LEAST ONE kernel has been replaced by
# a native ufunc; __native_functions__ names exactly which ones. A partial
# native module (only some kernels) leaves the rest as pure-Python.
__native_available__ = False
__native_module__ = None
__native_functions__ = ()


def _prefer_native() -> None:
    """Prefer compiled ufuncs when a sibling native extension is installed.

    Best-effort optimization: if the extension is absent, or present but fails
    to import (missing transitive dependency, ABI mismatch, bad artifact), fall
    back to the pure-Python kernels rather than breaking ``import ppspatial``.
    A present-but-broken extension warns; a simply-absent one is silent.
    """
    global __native_available__, __native_module__, __native_functions__

    try:
        native = _import_module("ppspatial_native")
    except ModuleNotFoundError as exc:
        if exc.name == "ppspatial_native":
            return  # not installed — expected, stay on pure Python silently
        _warn(
            f"ppspatial_native is present but a dependency is missing ({exc}); "
            "falling back to pure-Python kernels",
            RuntimeWarning,
        )
        return
    except Exception as exc:  # bad .so, ABI mismatch, init-time error, ...
        _warn(
            f"ppspatial_native is present but failed to import ({exc}); "
            "falling back to pure-Python kernels",
            RuntimeWarning,
        )
        return

    replaced = []
    for name in __all__:
        if hasattr(native, name):
            globals()[name] = getattr(native, name)
            replaced.append(name)

    if replaced:
        __native_available__ = True
        __native_module__ = native
        __native_functions__ = tuple(replaced)


_prefer_native()

del _prefer_native, _import_module, _warn
