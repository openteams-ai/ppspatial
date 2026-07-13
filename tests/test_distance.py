"""Tests for ppspatial distance kernels — validated against known values.

Reference values are hardcoded (scipy is never imported at runtime, per the
PostSciPy working rules). Distances are exact up to floating-point rounding,
so tolerances are tight.
"""

import math

from ppspatial import euclidean, sqeuclidean, cityblock, chebyshev


def close(a, b, rtol=1e-12, atol=1e-12):
    return abs(a - b) <= atol + rtol * abs(b)


# 3-4-5 right triangle: the classic exact reference.
A = [0.0, 0.0]
B = [3.0, 4.0]


class TestEuclidean:
    def test_known_value(self):
        assert close(euclidean(A, B), 5.0)

    def test_identity_is_zero(self):
        assert close(euclidean(B, B), 0.0)

    def test_symmetric(self):
        assert close(euclidean(A, B), euclidean(B, A))

    def test_higher_dim(self):
        # sqrt(1 + 4 + 9 + 16) = sqrt(30)
        p = [0.0, 0.0, 0.0, 0.0]
        q = [1.0, 2.0, 3.0, 4.0]
        assert close(euclidean(p, q), math.sqrt(30.0))


class TestSqeuclidean:
    def test_known_value(self):
        assert close(sqeuclidean(A, B), 25.0)

    def test_is_euclidean_squared(self):
        d = euclidean(A, B)
        assert close(sqeuclidean(A, B), d * d)

    def test_identity_is_zero(self):
        assert close(sqeuclidean(A, A), 0.0)


class TestCityblock:
    def test_known_value(self):
        # |3-0| + |4-0| = 7
        assert close(cityblock(A, B), 7.0)

    def test_symmetric(self):
        assert close(cityblock(A, B), cityblock(B, A))

    def test_negative_coords(self):
        # |(-1)-2| + |5-1| = 3 + 4 = 7
        assert close(cityblock([-1.0, 5.0], [2.0, 1.0]), 7.0)


class TestChebyshev:
    def test_known_value(self):
        # max(|3-0|, |4-0|) = 4
        assert close(chebyshev(A, B), 4.0)

    def test_identity_is_zero(self):
        assert close(chebyshev(B, B), 0.0)

    def test_picks_largest_axis(self):
        # differences: 1, 8, 3 -> max 8
        assert close(chebyshev([0.0, 0.0, 0.0], [1.0, 8.0, 3.0]), 8.0)


class TestOrdering:
    """Chebyshev <= Euclidean <= Cityblock for the same pair (L-inf <= L2 <= L1)."""

    def test_metric_ordering(self):
        p = [1.0, 2.0, 3.0]
        q = [4.0, 0.0, 1.0]
        assert chebyshev(p, q) <= euclidean(p, q) + 1e-12
        assert euclidean(p, q) <= cityblock(p, q) + 1e-12


class TestEdgeCases:
    ALL = [euclidean, sqeuclidean, cityblock, chebyshev]

    def test_identical_points_are_zero(self):
        p = [3.0, -4.0, 5.0]
        for fn in self.ALL:
            assert close(fn(p, p), 0.0)

    def test_single_element(self):
        # d=1: every metric reduces to |a-b| (euclidean/cityblock/chebyshev)
        # or (a-b)**2 (sqeuclidean)
        assert close(euclidean([7.0], [2.0]), 5.0)
        assert close(cityblock([7.0], [2.0]), 5.0)
        assert close(chebyshev([7.0], [2.0]), 5.0)
        assert close(sqeuclidean([7.0], [2.0]), 25.0)

    def test_all_negative_coords(self):
        # euclidean of (-1,-2,-3) vs (-4,-6,-8): sqrt(9+16+25)=sqrt(50)
        assert close(euclidean([-1.0, -2.0, -3.0], [-4.0, -6.0, -8.0]),
                     math.sqrt(50.0))
