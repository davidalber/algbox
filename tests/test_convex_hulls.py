from random import random, seed
from pylab import *
import sys

import unittest
import itertools
import numpy as np
from algbox.convex_hulls import ConvexHulls

class TestConvexHulls(unittest.TestCase):
    def test_hull_n3(self):
        """Test the convex hull algorithm with three point input."""
        points = np.array([[1, 1], [0, 0], [-1, 1]])
        ch = ConvexHulls(points)
        self.assertEquals(len(ch), 1)
        self.assertTrue(np.all([1, 0, 2] == ch[0]))

        ch = ConvexHulls(points)
        self.assertEquals(len(ch), 1, sys.maxint)
        self.assertTrue(np.all([1, 0, 2] == ch[0]))

        points = np.array([[2, 1], [0, 0], [1, 1]])
        ch = ConvexHulls(points)
        self.assertEquals(len(ch), 1)
        self.assertTrue(np.all([1, 0, 2] == ch[0]))

        ch = ConvexHulls(points)
        self.assertEquals(len(ch), 1, 5)
        self.assertTrue(np.all([1, 0, 2] == ch[0]))

    def test_hull_n4(self):
        """Test the convex hull algorithm with four point input."""
        # All point are in the hull.
        points = np.array([[1, 1], [0, 0], [-1, 1], [0.8, 0.5]])
        ch = ConvexHulls(points)
        self.assertEquals(len(ch), 1)
        self.assertTrue(np.all([1, 3, 0, 2] == ch[0]))

        ch = ConvexHulls(points, 5)
        self.assertEquals(len(ch), 1)
        self.assertTrue(np.all([1, 3, 0, 2] == ch[0]))

        # One of the points is not in the hull.
        points = np.array([[1, 1], [0, 0], [-1, 1], [0.2, 0.5]])
        ch = ConvexHulls(points)
        self.assertEquals(len(ch), 1)
        self.assertTrue(np.all([1, 0, 2] == ch[0]))

        ch = ConvexHulls(points, 5)
        self.assertEquals(len(ch), 1)
        self.assertTrue(np.all([1, 0, 2] == ch[0]))

    def test_hull_n25(self):
        n = 25
        seed(1)
        points = np.array([random() for i in range(2*n)]).reshape((n,2))
        expected = [[13, 23, 16, 20, 12, 10, 14, 19, 1], [2, 24, 17, 18, 6, 7],
                    [9, 8, 5, 0, 21, 22, 15], [3, 4, 11]]
        ## for i in range(n):
        ##     annotate(i, points[i])
        ## show()
        ch = ConvexHulls(points)
        self.assertEquals(len(ch), 1)
        self.assertTrue(np.all(expected[0] == ch[0]))

        ch = ConvexHulls(points, 2)
        self.assertEquals(len(ch), 2)
        for i in range(len(ch)):
            self.assertTrue(np.all(expected[i] == ch[i]))

    def test_len(self):
        """Test to verify statements like

            >>> ch = ConvexHull(points)
            >>> len(ch)

        behave as expected."""
        n = 25
        seed(1)
        expected = [[13, 23, 16, 20, 12, 10, 14, 19, 1], [2, 24, 17, 18, 6, 7],
                    [9, 8, 5, 0, 21, 22, 15], [3, 4, 11]]
        points = np.array([random() for i in range(2*n)]).reshape((n,2))

        ch = ConvexHulls(points)
        self.assertEquals(len(ch), 1)

        ch = ConvexHulls(points, 2)
        self.assertEquals(len(ch), 2)

        ch = ConvexHulls(points, sys.maxint)
        self.assertEquals(len(ch), 4)

    def test_getitem(self):
        """Test to verify statements like

            >>> ch = ConvexHull(points)
            >>> ch[0]

        behave as expected."""
        n = 25
        seed(1)
        points = np.array([random() for i in range(2*n)]).reshape((n,2))
        expected = [[13, 23, 16, 20, 12, 10, 14, 19, 1], [2, 24, 17, 18, 6, 7],
                    [9, 8, 5, 0, 21, 22, 15], [3, 4, 11]]

        ch = ConvexHulls(points)
        for i in range(len(ch)):
            for j in range(len(ch[i])):
                self.assertEquals(ch[i][j], expected[i][j])

        ch = ConvexHulls(points, 2)
        for i in range(len(ch)):
            for j in range(len(ch[i])):
                self.assertEquals(ch[i][j], expected[i][j])

        ch = ConvexHulls(points, 4)
        for i in range(len(ch)):
            for j in range(len(ch[i])):
                self.assertEquals(ch[i][j], expected[i][j])

        self.assertRaises(IndexError, ch.__getitem__, len(ch))
