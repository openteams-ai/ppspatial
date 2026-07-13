"""ppspatial — postpyc reimplementation of scipy.spatial.

Each function is a POST Python kernel written in fully-typed postpyc. The
compiler lowers the kernels to native shared-library code; in interpreted
mode they run via the pure-Python broadcast loop. When the optional
``ppspatial_native`` extension module is installed next to this package,
matching public functions are replaced with native NumPy ufuncs at import
time.

Function families implemented
------------------------------
Distance (_distance) : euclidean, sqeuclidean, cityblock, chebyshev

Roadmap (not yet implemented)
------------------------------
Distance             : minkowski, cosine, correlation, cdist
Transforms           : Rotation core ops on Shape[4] quaternions
Blocked on compiler  : pdist condensed form (computed core dim n(n-1)/2),
                       KDTree / ConvexHull / Delaunay
See ROADMAP.md for status and upstream postpython requests.
"""

from importlib import import_module as _import_module

from ppspatial._distance import (
    euclidean,
    sqeuclidean,
    cityblock,
    chebyshev,
)

__all__ = [
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
]

__native_available__ = False
__native_module__ = None


def _prefer_native() -> None:
    """Prefer compiled ufuncs when a sibling native extension is installed."""
    global __native_available__, __native_module__

    try:
        native = _import_module("ppspatial_native")
    except ModuleNotFoundError as exc:
        if exc.name == "ppspatial_native":
            return
        raise

    replaced = []
    for name in __all__:
        if hasattr(native, name):
            globals()[name] = getattr(native, name)
            replaced.append(name)

    if replaced:
        __native_available__ = True
        __native_module__ = native


_prefer_native()

del _prefer_native, _import_module
