import unittest
from collections import deque
from typing import List

from hypothesis import given
from hypothesis import strategies as st

from convex_hull import Point
from convex_hull import sort_clockwise
from convex_hull import compute_hull
from convex_hull import is_clockwise
from convex_hull import is_counter_clockwise
from convex_hull import y_intercept


class TestGivenFunctions(unittest.TestCase):
    """This class checks simple cases for the given functions."""

    def test_y_intercept(self):
        p1 = (0, 0)
        p2 = (20, 40)
        xs = [i for i in range(41)]
        for x in xs:
            y_int = y_intercept(p1, p2, x)
            self.assertAlmostEqual(2 * x, y_int, places=5)
        return

    def test_clockwise(self):
        p1 = (0, 0)
        p2 = (1, 0)
        p3 = (1, 1)

        self.assertTrue(is_clockwise(p1, p2, p3))
        self.assertFalse(is_clockwise(p1, p3, p2))
        return

    def test_counter_clockwise(self):
        p1 = (0, 0)
        p2 = (1, 0)
        p3 = (1, 1)

        self.assertTrue(is_counter_clockwise(p1, p3, p2))
        self.assertFalse(is_counter_clockwise(p1, p2, p3))
        return

    def test_clockwise_sort(self):
        p1 = (0, 0)
        p2 = (1, 0)
        p3 = (1, 1)
        p4 = (0, 1)
        points = [p2, p4, p1, p3]
        sort_clockwise(points)

        test_points = points + points[:2]
        for i in range(len(points)):
            a = test_points[i]
            b = test_points[i + 1]
            c = test_points[i + 2]
            self.assertTrue(is_clockwise(a, b, c))
        return


def is_convex_hull(hull: List[Point], points: List[Point]):
    vertices = hull + [hull[0]]
    prev_two = deque(maxlen=2)
    for vertex in vertices:
        prev_two.append(vertex)
        if len(prev_two) == 2:
            for point in points:
                assert not is_counter_clockwise(*prev_two, point)
    return True


class TestComputeHull(unittest.TestCase):
    """
    We provide one simple test here.
    You should write several specific tests for yourself.
    """

    @given(
        st.lists(  # generate a list
            st.tuples(  # of 2-tuples
                st.integers(
                    min_value=0, max_value=100_000
                ),  # of integers in the interval [0, 100_000]
                st.integers(min_value=0, max_value=100_000),
            ),
            min_size=3,  # minimum length of list
            max_size=100_000,  # maximum length of list
            unique=True,  # list will contain unique elements
        )
    )
    def test_compute_hull(self, points):
        points = list(points)
        sort_clockwise(points)

        hull = compute_hull(points)
        self.assertTrue(is_convex_hull(hull, points))
        return

    @given(
        st.lists(  # generate a list
            st.tuples(  # of 2-tuples
                st.integers(
                    min_value=0, max_value=100_000
                ),  # of integers in the interval [0, 100_000]
                st.integers(min_value=0, max_value=100_000),
            ),
            min_size=1,  # minimum length of list
            max_size=6,  # maximum length of list
            unique=True,  # list will contain unique elements
        )
    )
    def test_compute_hull_naive(self, points):
        """With <= 6 points, the program uses the 'naive' algorithm [O(n^3)]"""
        points = list(points)
        sort_clockwise(points)

        hull = compute_hull(points)
        self.assertTrue(is_convex_hull(hull, points))
        return

    def test_empty(self):
        """No points == no hull"""
        points = []
        hull = compute_hull(points)

        self.assertEqual(len(hull), 0)
        return

    def test_trivial_1_point(self):
        """If there is only one point, then that point is on the hull"""
        points = [(0, 1)]
        hull = compute_hull(points)

        self.assertTrue(is_convex_hull(hull, points))
        return

    def test_trivial_2_points(self):
        """If there are only two points, then both points are on the hull"""
        points = [(0, 1), (2, 3)]
        hull = compute_hull(points)

        self.assertTrue(is_convex_hull(hull, points))
        return

    def test_trivial_3_points(self):
        """If there are only three points, then all three points are on the hull"""
        points = [(5, 20), (10, 10), (15, 17)]
        hull = compute_hull(points)

        self.assertTrue(is_convex_hull(hull, points))
        return

    def test_hull_4_points(self):
        """If we align points in a triangle formation, then add one interior point,
        all points except the interior will be on the hull"""
        points = [(5, 20), (10, 10), (15, 17), (9, 15)]
        hull = compute_hull(points)

        self.assertTrue(is_convex_hull(hull, points))

        return

    def test_hull_square(self):
        """Test points with identical x/y coordinates"""
        points = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]
        hull = compute_hull(points)

        self.assertTrue(is_convex_hull(hull, points))

        return

    def test_hull_line(self):
        """Test completely collinear points"""
        points = [(i, i) for i in range(5)]
        hull = compute_hull(points)

        self.assertTrue(is_convex_hull(hull, points))

        return


if __name__ == "__main__":
    unittest.main()
    # test = TestComputeHull()
    # test.test_hull_square()
