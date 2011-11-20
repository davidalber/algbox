import unittest
import itertools
import math
import numpy as np
from algbox.delaunay import Delaunay

class TestDelaunay(unittest.TestCase):
    def correctness_tests(self, n, reps):
        for i in range(reps):
            d = Delaunay(n=n, seed=i)
            d.compute_triangulation()
            self.assertTrue(d.validate_triangulation())
            self.assertTrue(self.validate_through_flip_check(d))

    @staticmethod
    def validate_through_flip_check(d):
        """Validates a Delaunay triangulation by verifying that no flips need
        to be done. This is here as an alternative to the validation implementation
        in the Delaunay class.

        Returns True if validation passes."""
        for tri_id1, triangle1 in d.triangles.items():
            for edge in itertools.combinations(triangle1, 2):
                # Get the other triangle sharing this edge, if one exists.
                if len(d.edge_mapping[edge]) == 2:
                    angle1 = d.get_angle(triangle1, list(set(triangle1)-set(edge))[0])
                    tri_id2 = list(d.edge_mapping[edge].difference([tri_id1]))[0]
                    triangle2 = d.triangles[tri_id2]
                    angle2 = d.get_angle(triangle2, list(set(triangle2)-set(edge))[0])

                    if angle1+angle2 > math.pi:
                        return False
        return True

    def test_correctness_n3(self):
        self.correctness_tests(3, 500)
            
    def test_correctness_n4(self):
        self.correctness_tests(4, 400)
            
    def test_correctness_n5(self):
        self.correctness_tests(5, 400)
            
    def test_correctness_n6(self):
        self.correctness_tests(6, 300)
            
    def test_correctness_n7(self):
        self.correctness_tests(7, 250)
            
    def test_correctness_n10(self):
        self.correctness_tests(10, 100)
            
    def test_correctness_n30(self):
        self.correctness_tests(30, 30)
            
    def test_validate(self):
        for i in range(100):
            d = Delaunay(n=15, seed=i)
            d.compute_triangulation()
            self.assertTrue(d.validate_triangulation())
            self.assertTrue(self.validate_through_flip_check(d))

    def test_validate_edge_mapping(self):
        """Verify the edge mapping is correct after triangulation."""
        n = 25
        d = Delaunay(n=n, seed=50)
        d.compute_triangulation()
        tri_edge_count = np.zeros((n, n), dtype=np.int32)
        edge_mapping_count = tri_edge_count.copy()
        for edge, emap in d.edge_mapping.items():
            i = edge[0]
            j = edge[1]
            self.assertTrue(i < j)
            edge_mapping_count[i, j] += len(d.edge_mapping[edge])

        for triangle in d.triangles.values():
            for edge in itertools.combinations(triangle, 2):
                i = edge[0]
                j = edge[1]
                self.assertTrue(i < j)
                tri_edge_count[i, j] += 1

        self.assertTrue(np.all(tri_edge_count == edge_mapping_count))
        self.assertEquals(np.max(tri_edge_count), 2)
