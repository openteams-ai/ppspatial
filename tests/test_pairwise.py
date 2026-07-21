"""Tests for ppspatial pairwise (cdist) kernel — validated against known values.

Reference values are hardcoded (scipy is never imported at runtime, per the
PostSciPy working rules). Distances are exact up to floating-point rounding,
so tolerances are tight.
"""

import math

from ppspatial import cdist, euclidean


def close(a, b, rtol=1e-12, atol=1e-12):
    return abs(a - b) <= atol + rtol * abs(b)


class TestCdist:
    def test_single_pair(self):
        # 1x1 distance matrix: the 3-4-5 triangle
        r = cdist([[0.0, 0.0]], [[3.0, 4.0]])
        assert r.shape == (1, 1)
        assert close(r[0][0], 5.0)

    def test_matrix_values(self):
        a = [[0.0, 0.0], [1.0, 1.0]]
        b = [[3.0, 4.0], [0.0, 0.0]]
        r = cdist(a, b)
        assert r.shape == (2, 2)
        assert close(r[0][0], 5.0)              # (0,0)->(3,4)
        assert close(r[0][1], 0.0)              # (0,0)->(0,0)
        assert close(r[1][0], math.sqrt(13.0))  # (1,1)->(3,4)
        assert close(r[1][1], math.sqrt(2.0))   # (1,1)->(0,0)

    def test_rectangular_shape(self):
        # n != m: output is n x m
        a = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        b = [[0.0, 0.0], [0.0, 1.0]]
        r = cdist(a, b)
        assert r.shape == (3, 2)

    def test_agrees_with_euclidean(self):
        # every entry must equal the point-to-point euclidean of the rows
        a = [[1.0, 2.0, 3.0], [4.0, 0.0, 1.0]]
        b = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        r = cdist(a, b)
        for i in range(len(a)):
            for j in range(len(b)):
                assert close(r[i][j], euclidean(a[i], b[j]))

    def test_nan_propagates(self):
        nan = float("nan")
        r = cdist([[nan, 0.0]], [[0.0, 0.0]])
        assert math.isnan(r[0][0])
