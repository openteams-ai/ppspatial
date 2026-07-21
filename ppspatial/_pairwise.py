"""Pairwise distance functions — ppspatial._pairwise

Public API
----------
cdist(a, b)  — all pairwise Euclidean distances between two point sets

``cdist`` is a ``@guvectorize`` kernel over the layout signature
``(n,d),(m,d)->(n,m)``: an ``n``-point set and an ``m``-point set, each a stack
of ``d``-dimensional coordinate vectors, in; the full ``n x m`` matrix of
distances out. Both output core dimensions (``n``, ``m``) come directly from
the input shapes — no *computed* core dimension is needed — so this compiles
under the current reference compiler. (The condensed one-set form ``pdist``,
whose output length is ``n(n-1)/2``, does need a computed dim and is tracked
separately; see ROADMAP.md.)

Mirrors ``scipy.spatial.distance.cdist(a, b)`` with its default
``metric="euclidean"``. Results are exact up to floating-point rounding, so
tests compare against hardcoded reference values.
"""

from postyp import Array, Float64
from postpyc import guvectorize
from postpyc.math import sqrt


@guvectorize([], "(n,d),(m,d)->(n,m)")
def cdist(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    """Euclidean distance between every row of ``a`` and every row of ``b``.

    ``out[i][j]`` is the L2 distance between point ``a[i]`` and point ``b[j]``.
    Mirrors ``scipy.spatial.distance.cdist(a, b)`` (default euclidean metric).
    A NaN in any coordinate propagates into the affected matrix entries.
    """
    for i in range(len(a)):
        for j in range(len(b)):
            acc: Float64 = 0.0
            diff: Float64 = 0.0
            for c in range(len(a[i])):
                diff = a[i][c] - b[j][c]
                acc += diff * diff
            out[i][j] = sqrt(acc)
