import argparse
import numpy as np
from random import random, seed
import sys
from convex_hull import ConvexHull

from pylab import axis, matplotlib, plot, Polygon, show
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

class ConvexHulls(object):
    def __init__(self, points, max_hulls=1, plot_points=True):
        self.points = points
        self.plot_points = plot_points
        self.pids = np.arange(points.shape[0])
        self.convex_hulls = []
        interior_points = np.ones((self.points.shape[0]), dtype=np.bool)
        interior_points[:] = True

        while np.sum(interior_points) > 2 and len(self.convex_hulls) < max_hulls:
            ch = ConvexHull(points[interior_points])
            ipids = self.pids[interior_points]
            self.convex_hulls.append([ipids[id] for id in ch.convex_hull])
            for pid in ch.convex_hull:
                interior_points[ipids[pid]] = False

    def __len__(self):
        return len(self.convex_hulls)

    def __getitem__(self, index):
        return self.convex_hulls.__getitem__(index)

    @property
    def x(self):
        return self.points[:,0]

    @property
    def y(self):
        return self.points[:,1]

    def plot(self):
        """Plot the point field and convex hulls."""
        fig = plt.figure(figsize=(8,8))
        ax = plt.axes([0,0,1,1])
        if self.plot_points:
            plot(self.x, self.y, 'ok')  # plot points

        # Draw hulls.
        patches = []
        for hull in self.convex_hulls:
            hull_points = np.array([self.points[pid] for pid in hull])
            patches.append(Polygon(hull_points))

        colors = 100*np.random.rand(len(patches))
        collection = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.1, lw=2, linestyle='solid')
        collection.set_array(np.array(colors))
        ax.add_collection(collection)

        axis((0, 1, 0, 1))

def parse_input():
    parser = argparse.ArgumentParser(description='Compute convex hull of a random point field.')
    parser.add_argument('-n', '--npoints', dest='npoints', action='store',
                        default=20,
                        help='number of points in the randomly-generated point field (default: %(default)s)')
    parser.add_argument('--no-plot-points', dest='plot_points', action='store_false',
                        help='do not plot points')
    parser.add_argument('--num-hulls', dest='nhulls', action='store', default=1,
                        help="set the number of hulls to compute; set to 'all' to compute all possible hulls (default: %(default)s)")
    parser.add_argument('-s', '--seed', dest='seed', action='store', default=None,
                        help='set the random seed (allows for reproducibility)')
    return parser.parse_args()

def gen_random_points(n):
    """Generate random points in [0,1] x [0,1]."""
    return np.array([random() for i in range(2*n)]).reshape((n,2))

def convex_hulls():
    args = parse_input()

    if args.seed is not None:
        seed(int(args.seed))
    points = gen_random_points(int(args.npoints))
    if args.nhulls == 'all':
        nhulls = sys.maxint
    else:
        nhulls = int(args.nhulls)
    ch = ConvexHulls(points, nhulls, args.plot_points)
    ch.plot()
    show()
