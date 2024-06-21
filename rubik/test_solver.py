import unittest
from typing import List

import rubik
import solver


class TestSolver(unittest.TestCase):
    def assert_good_path(
            self,
            start: rubik.Position,
            end: rubik.Position,
            path: List[rubik.Permutation],
    ):
        current = start
        for move in path:
            current = rubik.perm_apply(move, current)
        self.assertEqual(current, end)

    def test_shortest_path_0(self):
        """ Length 0 path."""
        start = rubik.I
        end = rubik.I
        ans = solver.shortest_path(start, end)
        self.assertEqual(len(ans), 0)

    def test_shortest_path_1(self):
        """ Length 1 path."""
        start = rubik.I
        end = rubik.perm_apply(rubik.F, start)
        ans = solver.shortest_path(start, end)
        self.assertEqual(len(ans), 1)
        self.assertEqual(ans, [rubik.F])

    def test_shortest_path_2(self):
        """ Length 2 path."""
        start = rubik.I
        middle = rubik.perm_apply(rubik.F, start)
        end = rubik.perm_apply(rubik.L, middle)
        ans = solver.shortest_path(start, end)
        self.assertEqual(len(ans), 2)
        self.assertEqual(ans, [rubik.F, rubik.L])

    def test_shortest_path_3(self):
        """ Length 3 path."""
        start = rubik.I
        middle1 = rubik.perm_apply(rubik.F, start)
        middle2 = rubik.perm_apply(rubik.F, middle1)
        end = rubik.perm_apply(rubik.Li, middle2)
        ans = solver.shortest_path(start, end)
        self.assertEqual(len(ans), 3)
        self.assert_good_path(start, end, ans)

    def test_shortest_path_4(self):
        """ Length 4 path."""
        start = rubik.I
        middle1 = rubik.perm_apply(rubik.F, start)
        middle2 = rubik.perm_apply(rubik.L, middle1)
        middle3 = rubik.perm_apply(rubik.F, middle2)
        end = rubik.perm_apply(rubik.L, middle3)
        ans = solver.shortest_path(start, end)
        self.assertEqual(len(ans), 4)
        self.assert_good_path(start, end, ans)

    def test_shortest_path_14(self):
        """ Length 14 path."""
        start = (6, 7, 8, 20, 18, 19, 3, 4, 5, 16, 17, 15, 0, 1, 2, 14, 12, 13, 10, 11, 9, 21, 22, 23)
        end = rubik.I
        ans = solver.shortest_path(start, end)
        self.assertEqual(len(ans), 14)
        self.assert_good_path(start, end, ans)

    def test_shortest_path_bad(self):
        """ No solution."""
        start = (7, 8, 6, 20, 18, 19, 3, 4, 5, 16, 17, 15, 0, 1, 2, 14, 12, 13, 10, 11, 9, 21, 22, 23)
        end = rubik.I
        ans = solver.shortest_path(start, end)
        self.assertEqual(ans, None)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)
