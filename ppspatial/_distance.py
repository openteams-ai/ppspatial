"""Point-to-point distance functions — ppspatial._distance

Public API
----------
euclidean(a, b)      — L2 (straight-line) distance
sqeuclidean(a, b)    — squared L2 distance (no sqrt)
cityblock(a, b)      — L1 (Manhattan / taxicab) distance
chebyshev(a, b)      — L-infinity (maximum coordinate) distance
cosine(a, b)         — cosine distance: 1 - (a·b)/(‖a‖‖b‖)
correlation(a, b)    — correlation distance: 1 - Pearson r
minkowski(a, b, p)   — Lp distance, generalizing the L-family above

Each kernel is a ``@guvectorize`` function over the layout signature
``(d),(d)->()``: two 1-D coordinate vectors of equal length ``d`` in, one
scalar distance out. ``minkowski`` additionally takes a scalar order ``p``
(``(d),(d),()->()``). The runtime broadcasts each kernel over batches of
points automatically (the trailing axis is the core dimension), so
``euclidean(A, B)`` over stacks of vectors just works.

These mirror ``scipy.spatial.distance``. The results are exact (up to
floating-point rounding), not approximations, so tests compare against
hardcoded reference values with a tight tolerance.
"""

from postyp import Array, Float64
from postpyc import guvectorize
from postpyc.math import sqrt, fabs, pow


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


@guvectorize([], "(d),(d)->()")
def cosine(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    """Cosine distance: 1 - (a·b) / (‖a‖ ‖b‖).

    Zero for vectors pointing the same direction, up to 2 for opposite ones;
    it ignores magnitude and measures only the angle. Mirrors
    ``scipy.spatial.distance.cosine``. A zero-magnitude input divides by zero
    (inf/NaN), matching scipy.
    """
    dot: Float64 = 0.0
    na: Float64 = 0.0
    nb: Float64 = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    out[0] = 1.0 - dot / (sqrt(na) * sqrt(nb))


@guvectorize([], "(d),(d)->()")
def correlation(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    """Correlation distance: 1 - Pearson correlation of a and b.

    Equivalent to the cosine distance between the mean-centered vectors.
    Mirrors ``scipy.spatial.distance.correlation``. A constant input has zero
    centered magnitude and divides by zero (NaN), matching scipy.
    """
    amean: Float64 = 0.0
    bmean: Float64 = 0.0
    for i in range(len(a)):
        amean += a[i]
        bmean += b[i]
    amean = amean / len(a)
    bmean = bmean / len(a)
    dot: Float64 = 0.0
    na: Float64 = 0.0
    nb: Float64 = 0.0
    da: Float64 = 0.0
    db: Float64 = 0.0
    for i in range(len(a)):
        da = a[i] - amean
        db = b[i] - bmean
        dot += da * db
        na += da * da
        nb += db * db
    out[0] = 1.0 - dot / (sqrt(na) * sqrt(nb))


@guvectorize([], "(d),(d),()->()")
def minkowski(a: Array[Float64], b: Array[Float64], p: Float64,
              out: Array[Float64]) -> None:
    """Minkowski distance of order ``p``: (sum |a_i - b_i|**p) ** (1/p).

    Generalizes the L-family: ``p=1`` is ``cityblock`` (L1), ``p=2`` is
    ``euclidean`` (L2), and ``p -> inf`` approaches ``chebyshev`` (L-inf).
    Mirrors ``scipy.spatial.distance.minkowski``; ``p >= 1`` gives a true
    metric. NaN differences propagate through ``pow``. A non-positive ``p``
    divides by zero in the ``1/p`` exponent (inf/NaN), as in scipy.
    """
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += pow(fabs(a[i] - b[i]), p)
    out[0] = pow(acc, 1.0 / p)
