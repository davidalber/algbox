import numpy as np
import math

class ConvexHull(object):
    def __init__(self, points):
        self.points = points
        self.compute_hull()

    @property
    def x(self):
        return self.points[:,0]

    @property
    def y(self):
        return self.points[:,1]

    @property
    def hull(self):
        return self.convex_hull

    def compute_hull(self):
        """Compute the convex hull of the point field using Graham scan."""
        # Start with the hull containing the lowest point in the point field.
        self.convex_hull = [np.argmin(self.points[:,1], 0)]

        # Get the cosines of the angles of all lines between the "lowest" point
        # and all other points.
        ids_and_cos = np.array([[i,self.line_cosine(self.convex_hull[0], i)] for i in range(len(self.points)) if i != self.convex_hull[0]])

        # Get a sorted list of point ids for points furthest to right (by
        # angle) to furthest left, from the perspective of the lowest point.
        ids = ids_and_cos[np.argsort(ids_and_cos[:,1], 0)][::-1][:,0]

        # Now go through the sorted list
        self.convex_hull.append(ids[0])
        for pid in ids[1:]:
            while self.is_right_turn(pid):
                self.convex_hull.pop()
            self.convex_hull.append(pid)

        # Make sure the last point added does not introduce a right turn
        # to our first point.
        if self.is_right_turn(self.convex_hull[0]):
            self.convex_hull.pop()

        return np.array(self.convex_hull, dtype=np.int)

    def is_right_turn(self, id3):
        """Determines if point indicated by id is a "right turn" from the
        line made by the last two points in the convex hull."""
        id1 = self.convex_hull[-2]
        id2 = self.convex_hull[-1]
        cross_prod = (self.x[id2] - self.x[id1])*(self.y[id3] - self.y[id1]) - \
                     (self.y[id2] - self.y[id1])*(self.x[id3] - self.x[id1])
        return cross_prod < 0

    def line_cosine(self, id1, id2):
        """Computes the cosine of the angle between the x-axis and the line
        between the points referred to by id1 and id2."""
        return (self.x[id2] - self.x[id1]) / self.point_distance(self.points[id1],
                                                                 self.points[id2])

    @staticmethod
    def point_distance(p1, p2):
        """Computes the distance between two points."""
        return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))
