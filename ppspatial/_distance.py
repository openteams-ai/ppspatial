"""Point-to-point distance functions — ppspatial._distance

Public API
----------
euclidean(a, b)     — L2 (straight-line) distance
sqeuclidean(a, b)   — squared L2 distance (no sqrt)
cityblock(a, b)     — L1 (Manhattan / taxicab) distance
chebyshev(a, b)     — L-infinity (maximum coordinate) distance

Each kernel is a ``@guvectorize`` function over the layout signature
``(d),(d)->()``: two 1-D coordinate vectors of equal length ``d`` in, one
scalar distance out. The runtime broadcasts each kernel over batches of
points automatically (the trailing axis is the core dimension), so
``euclidean(A, B)`` over stacks of vectors just works.

These mirror ``scipy.spatial.distance``. The results are exact (up to
floating-point rounding), not approximations, so tests compare against
hardcoded reference values with a tight tolerance.
"""

from postyp import Array, Float64
from postpyc import guvectorize
from postpyc.math import sqrt, fabs


@guvectorize([], "(d),(d)->()")
def euclidean(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    """Euclidean (L2) distance: sqrt(sum((a[i] - b[i])**2))."""
    acc: Float64 = 0.0
    diff: Float64 = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        acc += diff * diff
    out[0] = sqrt(acc)


@guvectorize([], "(d),(d)->()")
def sqeuclidean(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    """Squared Euclidean distance: sum((a[i] - b[i])**2).

    Cheaper than ``euclidean`` (no sqrt) and monotonic in it, so it is the
    preferred metric when only relative ordering of distances matters.
    """
    acc: Float64 = 0.0
    diff: Float64 = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        acc += diff * diff
    out[0] = acc


@guvectorize([], "(d),(d)->()")
def cityblock(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    """City-block / Manhattan (L1) distance: sum(|a[i] - b[i]|)."""
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += fabs(a[i] - b[i])
    out[0] = acc


@guvectorize([], "(d),(d)->()")
def chebyshev(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    """Chebyshev (L-infinity) distance: max(|a[i] - b[i]|).

    NaN propagates (matching numpy/scipy): once a NaN difference is seen the
    result is NaN. This needs an explicit guard because `NaN > m` is always
    False, so the running max would otherwise silently ignore NaN.
    """
    m: Float64 = 0.0
    d: Float64 = 0.0
    for i in range(len(a)):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
        if d != d:  # d is NaN; force the result to NaN (NaN > m never fires)
            m = d
    out[0] = m
