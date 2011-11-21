from random import random, seed
from pylab import *

import unittest
import itertools
import numpy as np
from algbox.convex_hull import ConvexHull

class TestConvexHull(unittest.TestCase):
    def test_hull_n3(self):
        """Test the convex hull algorithm with three point input."""
        points = np.array([[1, 1], [0, 0], [-1, 1]])
        ch = ConvexHull(points)
        self.assertTrue(np.all([1, 0, 2] == ch.hull))

        points = np.array([[2, 1], [0, 0], [1, 1]])
        ch = ConvexHull(points)
        self.assertTrue(np.all([1, 0, 2] == ch.hull))

    def test_hull_n4(self):
        """Test the convex hull algorithm with four point input."""
        # All point are in the hull.
        points = np.array([[1, 1], [0, 0], [-1, 1], [0.8, 0.5]])
        ch = ConvexHull(points)
        self.assertTrue(np.all([1, 3, 0, 2] == ch.hull))

        # One of the points is not in the hull.
        points = np.array([[1, 1], [0, 0], [-1, 1], [0.2, 0.5]])
        ch = ConvexHull(points)
        self.assertTrue(np.all([1, 0, 2] == ch.hull))

    def test_hull_n150(self):
        n = 25
        seed(1)
        points = np.array([random() for i in range(2*n)]).reshape((n,2))
        ch = ConvexHull(points)
        ## for i in range(n):
        ##     annotate(i, points[i])
        ## show()
        self.assertTrue(np.all([13, 23, 16, 20, 12, 10, 14, 19, 1] == ch.hull))

    def test_is_right_turn(self):
        """Test the is_right_turn method."""
        points = np.array([[1, 1], [0, 0], [-1, 1], [0.8, 0.5]])
        ch = ConvexHull(points)
        ch.convex_hull = [1, 2]
        self.assertTrue(ch.is_right_turn(0))
        self.assertTrue(ch.is_right_turn(3))

        ch.convex_hull = [0, 1]
        self.assertTrue(ch.is_right_turn(2))
        self.assertFalse(ch.is_right_turn(3))

        ch.convex_hull = [1, 0]
        self.assertFalse(ch.is_right_turn(2))
        self.assertTrue(ch.is_right_turn(3))

    def test_line_cosine(self):
        """Test the line_cosine method."""
        points = np.array([[1, 1], [0, 0], [-1, 1], [0.8, 0.5]])
        ch = ConvexHull(points)
        self.assertTrue(np.allclose(ch.line_cosine(0, 1), math.cos(5*math.pi/4)))
        self.assertTrue(np.allclose(ch.line_cosine(0, 2), math.cos(math.pi)))
        self.assertTrue(np.allclose(ch.line_cosine(1, 3), math.cos(0.5585993153435626)))

    def test_point_distance(self):
        """Test the point_distance method."""
        p1 = np.array([0, 0])
        p2 = np.array([0, 7])
        self.assertEquals(ConvexHull.point_distance(p1, p2), 7)

        p1 = np.array([3.5, 7.4])
        p2 = np.array([-6, -9.2])
        self.assertEquals(ConvexHull.point_distance(p1, p2), 19.126160095534075)

    def test_contains(self):
        """Test to verify that statements like

            >>> ch = ConvexHull(points)
            >>> 3 in ch

        behave as expected."""
        n = 25
        seed(1)
        points = np.array([random() for i in range(2*n)]).reshape((n,2))
        ch = ConvexHull(points)
        expected = [13, 23, 16, 20, 12, 10, 14, 19, 1]
        self.assertTrue(np.all(expected == ch.hull))
        for exp in expected:
            self.assertTrue(exp in ch)
        self.assertFalse(15 in ch)
        self.assertFalse(25 in ch)

    def test_len(self):
        """Test to verify statements like

            >>> ch = ConvexHull(points)
            >>> len(ch)

        behave as expected."""
        n = 25
        seed(1)
        points = np.array([random() for i in range(2*n)]).reshape((n,2))
        ch = ConvexHull(points)
        self.assertTrue(np.all([13, 23, 16, 20, 12, 10, 14, 19, 1] == ch.hull))
        self.assertEquals(len(ch), 9)

    def test_getitem(self):
        """Test to verify statements like

            >>> ch = ConvexHull(points)
            >>> ch[0]

        behave as expected."""
        n = 25
        seed(1)
        points = np.array([random() for i in range(2*n)]).reshape((n,2))
        ch = ConvexHull(points)
        expected = [13, 23, 16, 20, 12, 10, 14, 19, 1]
        self.assertTrue(np.all(expected == ch.hull))
        for i in range(len(ch)):
            self.assertEquals(ch[i], expected[i])
        self.assertRaises(IndexError, ch.__getitem__, len(ch))
