from pylab import *
import numpy as np
from random import random
import itertools
import math

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

class Delaunay(object):
    def __init__(self, n):
        self.gen_random_points(n)

    def gen_random_points(self, n):
        """Generate random points in [0,1] x [0,1]."""
        self.points = np.array([random() for i in range(2*n)]).reshape((n,2))

    @property
    def x(self):
        return self.points[:,0]

    @property
    def y(self):
        return self.points[:,1]

    def triangulation(self):
        convex_hull = self.get_convex_hull()
        self.convex_hull = convex_hull
        # Find point not in convex hull boundary and then use it to make
        # triangles with the convex hull boundary points.
        for middle_point in range(len(self.points)):
            if middle_point not in convex_hull:
                break

        self.triangles = []
        for h1,h2 in zip(convex_hull, np.concatenate((convex_hull[1:],
                                                      [convex_hull[0]]))):
            self.triangles.append([h1, h2, middle_point])
        

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
            self.convex_hull.pop()

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
        fig = plt.figure(figsize=(5,5))
        ax = plt.axes([0,0,1,1])        
        scatter(self.x, self.y)

        # Draw convex hull.
        for i in range(len(self.convex_hull)-1):
            plot([self.x[self.convex_hull[i]], self.x[self.convex_hull[i+1]]],
                 [self.y[self.convex_hull[i]], self.y[self.convex_hull[i+1]]],
                 'r', lw=4)
        plot([self.x[self.convex_hull[0]], self.x[self.convex_hull[-1]]],
             [self.y[self.convex_hull[0]], self.y[self.convex_hull[-1]]],
             'r', lw=4)

        # Draw triangles.
        patches = []
        for triangle in self.triangles:
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
        
        return mpatches.Circle([center_x, center_y], radius, ec="none")

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

d = Delaunay(20)
d.triangulation()
d.plot()
show()
