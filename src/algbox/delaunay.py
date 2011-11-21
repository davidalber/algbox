import argparse
from pylab import *
import numpy as np
import random
import itertools
import math
from algbox.convex_hull import ConvexHull

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

class Delaunay(object):
    def __init__(self, n, seed=None, summary=False, verbose=False):
        if seed is not None:
            random.seed(seed)
        self.points = self.gen_random_points(n)
        self.summary = summary
        if summary:
            self.ntriangles = 0
            self.nadds = 0
            self.nremoves = 0
            self.nflips = 0
        self.verbose = verbose

    def gen_random_points(self, n):
        """Generate random points in [0,1] x [0,1]."""
        return np.array([random.random() for i in range(2*n)]).reshape((n,2))

    @property
    def x(self):
        return self.points[:,0]

    @property
    def y(self):
        return self.points[:,1]

    def compute_triangulation(self):
        chull = ConvexHull(self.points)
        self.convex_hull = chull.hull

        self.triangle_id = 0
        self.triangles = {}
        self.edge_mapping = {}

        self.initial_triangles_from_hull()

        # Add remaining points incrementally.
        remaining_points = [i for i in range(len(self.points))
                            if i not in self.convex_hull]

        for p in remaining_points:
            self._accounting('add_point', p)
            in_triangle = self.point_in_triangle(p)
            self.split_triangle(in_triangle, p)

    def split_triangle(self, tri_id, point_id):
        """Split the triangle with ID tri_id into three triangles using the
        given point_id."""
        parent_triangle = self.triangles[tri_id]
        self._accounting('split', parent_triangle)
        if self.summary:
            self.nremoves += 1
            self.ntriangles -= 1
        self.remove_triangle(tri_id)
        for a,b in itertools.combinations(parent_triangle, 2):
            new_triangle = sort([a, b, point_id])
            self._accounting('add_triangle', new_triangle)
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

    def _accounting(self, event, data=None):
        """This method is used to print and record information about the
        triangulation. It only has effect when the user requests verbose
        output or the summary."""
        if not self.summary and not self.verbose:
            return
        if event == 'add_point':
            if self.verbose:
                print 'Adding point {}'.format(data)
        elif event == 'add_triangle':
            if self.verbose:
                print '    Adding triangle {}'.format(data)
            if self.summary:
                self.nadds += 1
                self.ntriangles +=1
        elif event == 'build_hull':
            if self.verbose:
                print 'Building convex hull... {}'.format(data)
        elif event == 'create_initial':
            if self.verbose:
                print 'Creating initial triangle {}'.format(data)
            if self.summary:
                self.nadds += 1
                self.ntriangles += 1
        elif event == 'flip':
            if self.verbose:
                print '    Flipping triangles {} and {}'.format(data[0], data[1])
            if self.summary:
                self.nflips += 1
        elif event == 'select_interior':
            if self.verbose:
                print 'Selecting starting interior point... {}'.format(data)
        elif event == 'split':
            if self.verbose:
                print 'Splitting triangle'
                print '    Removing triangle {}'.format(data)
        else:
            raise ValueError('unknown event')

    def initial_triangles_from_hull(self):
        # Triangulate the convex hull.
        start_point = self.convex_hull[0]
        shifted_hull = np.concatenate((self.convex_hull[2:], [self.convex_hull[1]]))
        for h1,h2 in zip(self.convex_hull[1:], shifted_hull)[:-1]:
            tri_edges = sort([h1, h2, start_point])
            self._accounting('create_initial', tri_edges)
            self.add_triangle(tri_edges)
            for key in itertools.combinations(tri_edges, 2):
                if self.check_and_flip(key):
                    break

    def check_and_flip(self, shared_edge):
        """Check an adjacent pair of triangles and flip if necessary."""
        if len(self.edge_mapping[shared_edge]) == 2:
            id1, id2 = list(self.edge_mapping[shared_edge])
            tri1 = self.triangles[id1]
            angle1 = self.get_angle(tri1, list(set(tri1)-set(shared_edge))[0])
            tri2 = self.triangles[id2]
            angle2 = self.get_angle(tri2, list(set(tri2)-set(shared_edge))[0])

            if angle1+angle2 > math.pi:
                self._accounting('flip', [tri1, tri2])
                self.flip(id1, id2)

                # Check flipped triangles for the Delaunay condition and
                # flip as necessary.
                for key in itertools.combinations(self.triangles[id1], 2):
                    if self.check_and_flip(key):
                        break

                for key in itertools.combinations(self.triangles[id2], 2):
                    if self.check_and_flip(key):
                        break

                return True
        return False

    def flip(self, id1, id2):
        """Flip the common edge between two adjacent triangles.

        This should only be called for pairs of triangles where all four
        points are in the convex hull formed by the triangles' points.
        This constraint is not an issue for calls from method
        check_and_flip because triangle pairs where only three points
        are in the convex hull do not violate the opposite angle
        condition (i.e., the sum of the triangle corner angles opposite
        the common edge are guaranteed to be less then 180 degrees
        when only three points are in the convex hull)."""
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

        # Update edge_mapping, replacing triangle 2 with triangle 1
        # (and vice versa) for edges that were transferred between
        # triangles.
        for key in itertools.combinations(new_tri1, 2):
            if key != tuple(new_edge):
                self.edge_mapping[key].difference_update([id2])
                self.edge_mapping[key].add(id1)

        for key in itertools.combinations(new_tri2, 2):
            if key != tuple(new_edge):
                self.edge_mapping[key].difference_update([id1])
                self.edge_mapping[key].add(id2)

    def get_angle(self, triangle, corner):
        """Find the angle of the given corner of the given triangle."""
        non_corner = list(set(triangle) - set([corner]))
        # a is the edge opposite the corner
        a = self.point_distance(self.points[non_corner[0]], self.points[non_corner[1]])

        b = self.point_distance(self.points[corner], self.points[non_corner[0]])
        c = self.point_distance(self.points[corner], self.points[non_corner[1]])

        return math.acos((math.pow(b, 2) + math.pow(c, 2) - math.pow(a, 2)) / (2*b*c))

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
        for tri_id, triangle in self.triangles.iteritems():
            [center_x, center_y], radius = self.get_circumcircle(triangle)
            outside_point_ids = [pid for pid in range(len(self.points))
                                 if pid not in triangle]
            for pid in outside_point_ids:
                dist = self.point_distance([center_x, center_y], self.points[pid])
                if dist < radius:
                    return False
        return True

def parse_input():
    parser = argparse.ArgumentParser(description='Compute Delaunay triangulation on a random point field.')
    parser.add_argument('-n', '--npoints', dest='npoints', action='store',
                        default=20,
                        help='number of points in the randomly-generated point field (default: %(default)s)')
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        default=True, help='suppress plotting the triangulation')
    parser.add_argument('-s', '--seed', dest='seed', action='store', default=None,
                        help='set the random seed (allows for reproducibility)')
    parser.add_argument('--summary', dest='summary', action='store_true',
                        default=False,
                        help='compile triangulation summary information')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        default=False,
                        help='print information as the triangulation proceeds')
    parser.add_argument('--validate', dest='validate', action='store_true',
                        default=False,
                        help='validate the computed triangulation is Delaunay')
    return parser.parse_args()

def delaunay():
    args = parse_input()

    if args.seed is not None:
        d = Delaunay(int(args.npoints), seed=int(args.seed), summary=args.summary,
                     verbose=args.verbose)
    else:
        d = Delaunay(int(args.npoints), verbose=args.verbose, summary=args.summary)
    d.compute_triangulation()
    if args.summary:
        summary_vals = [str(v) for v in [d.points.shape[0], d.ntriangles,
                                         d.nadds, d.nremoves, d.nflips]]
        max_val_len = max([len(v) for v in summary_vals])
        summary_text = ['points', 'triangles', 'triangles added',
                        'triangles removed', 'triangle flips']
        min_text_len = min([len(t) for t in summary_text])
        print '\n============= Summary ============='
        for val, text in zip(summary_vals, summary_text):
            print 'Number of {} {} {}{}'.format(text,
                                                '.'*(14+min_text_len-len(text)),
                                                ' '*(max_val_len-len(val)),
                                                val)
    if args.validate:
        if d.validate_triangulation():
            print "Validation: is Delaunay!"
        else:
            print "Validation: is not Delaunay!"
    if args.plot:
        d.plot()
        show()
