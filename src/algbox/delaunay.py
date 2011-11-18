from pylab import *
import numpy as np
import random
import itertools
import math

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

class Delaunay(object):
    def __init__(self, n, seed=None):
        if seed is not None:
            random.seed(seed)
        self.gen_random_points(n)

    def gen_random_points(self, n):
        """Generate random points in [0,1] x [0,1]."""
        self.points = np.array([random.random() for i in range(2*n)]).reshape((n,2))

    @property
    def x(self):
        return self.points[:,0]

    @property
    def y(self):
        return self.points[:,1]

    def triangulation(self):
        self.convex_hull = self.get_convex_hull()

        self.triangle_id = 0
        self.triangles = {}
        self.edge_mapping = {}

        middle_point = self.initial_triangles_from_hull()

        # Add remaining points incrementally.
        remaining_points = [i for i in range(len(self.points))
                            if i not in self.convex_hull and i != middle_point]

        for p in remaining_points:
            in_triangle = self.point_in_triangle(p)
            self.split_triangle(in_triangle, p)

    def split_triangle(self, tri_id, point_id):
        """Split the triangle with ID tri_id into three triangles using the
        given point_id."""
        parent_triangle = self.triangles[tri_id]
        self.remove_triangle(tri_id)
        for a,b in itertools.combinations(parent_triangle, 2):
            new_triangle = sort([a, b, point_id])
            self.add_triangle(new_triangle)
            for key in itertools.combinations(new_triangle, 2):
                if self.check_and_flip(key):
                    break

    def point_in_triangle(self, p):
        """Determine the triangle that p is inside by searching for
        the triangle ABC for which the sum of the angles ApB, BpC,
        and CpA is 2*pi."""
        for id,triangle in self.triangles.iteritems():
            angle_sum = 0
            for a,b in itertools.combinations(triangle, 2):
                temp_triangle = [a, b, p]
                angle_sum += self.get_angle(temp_triangle, p)
            if np.allclose([angle_sum], [2*math.pi], 1e-10, 0):
                return id

    def add_triangle(self, corners):
        """Add new triangle with given corners."""
        tri_id = self.triangle_id
        self.triangle_id += 1
        self.triangles[tri_id] = corners
        for key in itertools.combinations(corners, 2):
            if self.edge_mapping.has_key(key):
                self.edge_mapping[key].add(tri_id)
            else:
                self.edge_mapping[key] = set([tri_id])
        return tri_id

    def remove_triangle(self, id):
        """Remove the triangle with the given ID."""
        corners = self.triangles[id]
        for key in itertools.combinations(corners, 2):
            self.edge_mapping[key].difference_update([id])
            if len(self.edge_mapping[key]) == 0:
                del(self.edge_mapping[key])
        del(self.triangles[id])

    def initial_triangles_from_hull(self):
        # Find point not in convex hull boundary and then use it to make
        # triangles with the convex hull boundary points.
        for middle_point in range(len(self.points)):
            if middle_point not in self.convex_hull:
                break

        for h1,h2 in zip(self.convex_hull, np.concatenate((self.convex_hull[1:],
                                                           [self.convex_hull[0]]))):
            tri_edges = sort([h1, h2, middle_point])
            self.add_triangle(tri_edges)
            for key in itertools.combinations(tri_edges, 2):
                if self.check_and_flip(key):
                    break

        #print 'after!'

        for tri_id, triangle in self.triangles.iteritems():
            for key in itertools.combinations(triangle, 2):
                # Get the other triangle sharing this edge, if one exists.
                if len(self.edge_mapping[key]) == 2:
                    angle1 = self.get_angle(triangle, list(set(triangle)-set(key))[0])
                    tri_id2 = list(self.edge_mapping[key].difference([tri_id]))[0]
                    triangle2 = self.triangles[tri_id2]
                    angle2 = self.get_angle(triangle2, list(set(triangle2)-set(key))[0])

                    #if angle1+angle2 > math.pi:
                        #print '!!! {}'.format(angle1+angle2)
        return middle_point

    def check_and_flip(self, shared_edge):
        """Check an adjacent pair of triangles and flip if necessary."""
        if len(self.edge_mapping[shared_edge]) == 2:
            id1 = list(self.edge_mapping[shared_edge])[0]
            id2 = list(self.edge_mapping[shared_edge])[1]
            tri1 = self.triangles[id1]
            angle1 = self.get_angle(tri1, list(set(tri1)-set(shared_edge))[0])
            tri2 = self.triangles[id2]
            angle2 = self.get_angle(tri2, list(set(tri2)-set(shared_edge))[0])

            if angle1+angle2 > math.pi:
                self.flip(id1, id2)
                #print '@@@ {}'.format(angle1+angle2)
                return True
        return False

    def flip(self, id1, id2):
        """Flip the common edge between two adjacent triangles."""
        tri1 = self.triangles[id1]
        tri2 = self.triangles[id2]
        shared_edge = sort(list(set(tri1).intersection(tri2)))
        new_edge = sort(list(set(np.concatenate((tri1, tri2))).difference(shared_edge)))

        new_tri1 = sort(np.concatenate((new_edge, [shared_edge[0]])))
        new_tri2 = sort(np.concatenate((new_edge, [shared_edge[1]])))

        self.edge_mapping[tuple(new_edge)] = self.edge_mapping[tuple(shared_edge)]
        del(self.edge_mapping[tuple(shared_edge)])

        self.triangles[id1] = new_tri1
        self.triangles[id2] = new_tri2

        for key in itertools.combinations(new_tri1, 2):
            if key != tuple(new_edge):
                self.edge_mapping[key].difference_update([id2])
                self.edge_mapping[key].add(id1)

        for key in itertools.combinations(new_tri2, 2):
            if key != tuple(new_edge):
                self.edge_mapping[key].difference_update([id1])
                self.edge_mapping[key].add(id2)

        for key in itertools.combinations(self.triangles[id1], 2):
            if self.check_and_flip(key):
                break

        for key in itertools.combinations(self.triangles[id2], 2):
            if self.check_and_flip(key):
                break

    def get_angle(self, triangle, corner):
        """Find the angle of the given corner of the given triangle."""
        non_corner = list(set(triangle) - set([corner]))
        # a is the edge opposite the corner
        a = self.point_distance(self.points[non_corner[0]], self.points[non_corner[1]])

        b = self.point_distance(self.points[corner], self.points[non_corner[0]])
        c = self.point_distance(self.points[corner], self.points[non_corner[1]])

        return math.acos((math.pow(b, 2) + math.pow(c, 2) - math.pow(a, 2)) / (2*b*c))

    def get_convex_hull(self):
        """Compute the convex hull of the point cloud using Graham scan."""
        # Start with the hull containing the lowest point in the point cloud.
        convex_hull = [np.argmin(self.points[:,1], 0)]

        # Get the cosines of the angles of all lines between the "lowest" point
        # and all other points.
        ids_and_cos = np.array([[i,self.line_cosine(convex_hull[0], i)] for i in range(len(self.points)) if i != convex_hull[0]])

        # Get a sorted list of point ids for points furthest to right (by
        # angle) to furthest left, from the perspective of the lowest point.
        ids = ids_and_cos[np.argsort(ids_and_cos[:,1], 0)][::-1][:,0]

        # Now go through the sorted list
        convex_hull.append(ids[0])
        for pid in ids[1:]:
            while self.is_right_turn(convex_hull, pid):
                convex_hull.pop()
            convex_hull.append(pid)

        # Make sure the last point added does not introduce a right turn
        # to our first point.
        if self.is_right_turn(convex_hull, convex_hull[0]):
            convex_hull.pop()

        return np.array(convex_hull, dtype=np.int)

    def is_right_turn(self, convex_hull, id3):
        """Determines if point indicated by id is a "right turn" from the
        line made by the last two points in the convex hull."""
        id1 = convex_hull[-2]
        id2 = convex_hull[-1]
        cross_prod = (self.x[id2] - self.x[id1])*(self.y[id3] - self.y[id1]) - \
                     (self.y[id2] - self.y[id1])*(self.x[id3] - self.x[id1])
        return cross_prod < 0

    def line_cosine(self, id1, id2):
        return (self.x[id2] - self.x[id1]) / self.point_distance(self.points[id1],
                                                                 self.points[id2])

    @staticmethod
    def point_distance(p1, p2):
        """Computes the distance between two points."""
        return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

    def plot(self):
        fig = plt.figure(figsize=(8,8))
        ax = plt.axes([0,0,1,1])        
        scatter(self.x, self.y)

        # Draw triangles.
        patches = []
        for triangle in self.triangles.itervalues():
            patches.append(self.draw_circumcircle(triangle))
            for i1,i2 in itertools.combinations(range(3), 2):
                p1 = triangle[i1]
                p2 = triangle[i2]
                plot([self.x[p1], self.x[p2]], [self.y[p1], self.y[p2]], 'k')

        colors = 100*np.random.rand(len(patches))
        collection = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.1)
        collection.set_array(np.array(colors))
        ax.add_collection(collection)
        axis((0, 1, 0, 1))

    def draw_circumcircle(self, triangle):
        [center_x, center_y], radius = self.get_circumcircle(triangle)
        return mpatches.Circle([center_x, center_y], radius, ec="none")

    def get_circumcircle(self, triangle):
        """Return the center and radius of the given triangle's
        circumcircle."""
        # Get the midpoints of two of the triangle's edges.
        mp1x,mp1y = self.get_midpoint(triangle[0], triangle[1])
        mp1_slope = -1./self.get_slope(triangle[0], triangle[1])
        mp1_intercept = self.get_intercept(mp1_slope, mp1x, mp1y)

        mp2x,mp2y = self.get_midpoint(triangle[1], triangle[2])
        mp2_slope = -1./self.get_slope(triangle[1], triangle[2])
        mp2_intercept = self.get_intercept(mp2_slope, mp2x, mp2y)

        # Compute intersection of the midpoint orthogonal lines.
        center_x = (mp2_intercept - mp1_intercept) / (mp1_slope - mp2_slope)
        center_y = mp1y + mp1_slope * (center_x - mp1x)
        radius = self.point_distance([center_x, center_y], self.points[triangle[0]])

        return [center_x, center_y], radius

    def get_intercept(self, slope, x, y):
        return y - slope*x

    def get_slope(self, id1, id2):
        rise = self.y[id1] - self.y[id2]
        run = self.x[id1] - self.x[id2]
        return rise/run

    def get_midpoint(self, id1, id2):
        mpx = (self.x[id1] + self.x[id2]) / 2
        mpy = (self.y[id1] + self.y[id2]) / 2
        return mpx,mpy

    def validate_triangulation(self):
        """Go through the triangles and validate that the triangulation
        is Delaunay."""
        print "Validation:"
        for tri_id, triangle in self.triangles.iteritems():
            [center_x, center_y], radius = self.get_circumcircle(triangle)
            outside_point_ids = [pid for pid in range(len(self.points))
                                 if pid not in triangle]
            for pid in outside_point_ids:
                dist = self.point_distance([center_x, center_y], self.points[pid])
                if dist < radius:
                    raise "Not Delaunay"
        print "\tIs Delaunay!"

        ##     for key in itertools.combinations(triangle, 2):
        ##         # Get the other triangle sharing this edge, if one exists.
        ##         if len(self.edge_mapping[key]) == 2:
        ##             angle1 = self.get_angle(triangle, list(set(triangle)-set(key))[0])
        ##             tri_id2 = list(self.edge_mapping[key].difference([tri_id]))[0]
        ##             triangle2 = self.triangles[tri_id2]
        ##             angle2 = self.get_angle(triangle2, list(set(triangle2)-set(key))[0])

        ##             if angle1+angle2 > math.pi:
        ##                 print "!!!\t{}".format(angle1+angle2)
        ##                 raise "Not Delaunay"
        ##             else:
        ##                 print "\t{}".format(angle1+angle2)
        ## print "\tIs Delaunay!"

def delaunay():
    d = Delaunay(100)
    d.triangulation()
    d.validate_triangulation()
    d.plot()
    show()

if __name__ == '__main__':
    delaunay()
