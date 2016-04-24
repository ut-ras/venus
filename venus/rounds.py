# In this step, we need to divide the image into disjoint sets. Each set
# must satisfy the following properites:
#       - contain at most N colors, where N is the number of cans
#       - for each color, have at most (max pixels per can) pixels
# The ideal solution will form a set that maximizes the number of pixels
# sprayed and minimize distance between pixels. It's important to create
# a set of adjacent pixels rather than a sparse array.

# Inputs:
#       - pixels                numpy.ndarray of Integers
#       - max pixels per can    Integer
#       - number of cans        Integer
# Outputs:
#       - set(([(x, y, can number)], [(color, can number)]))

import random
import argparse
import numpy as np
from scipy import spatial
from sklearn.cluster import MiniBatchKMeans
import pdb


class Cluster(object):
    def __init__(self, elems):
        self._donatable = set(elems)
        self._frozen = set()

    def __len__(self):
        return len(self._donatable) + len(self._frozen)

    def donation_candidates(self, centers):
        clus = np.array(list(self._donatable))
        tree = spatial.KDTree(centers)
        dists, target = tree.query(clus)
        by_dist = np.argsort(dists)
        return zip(clus[by_dist], target[by_dist])

    def remove(self, elem):
        self._donatable.remove(elem)

    def add(self, elem):
        self._frozen.add(elem)

    def count_donatable(self):
        return len(self._donatable)

    def all(self):
        return np.concatenate((
                np.array(list(self._donatable)),
                np.array(list(self._frozen))
                ))


def redistribute_clusters(clusters, centers, cluster_size, max_iterations=1000):
    clusters = [Cluster(c) for c in clusters]
    odd_size = sum(map(len, clusters)) % cluster_size

    by_size = np.argsort(np.array([len(c) for c in clusters]))
    smallest = np.argmin(np.abs(by_size - cluster_size))

    # Form the smallest cluster by giving away pixels from it
    mask = np.zeros(2 * len(centers), dtype=bool).reshape((-1, 2))
    mask[smallest] = True
    centers = np.ma.masked_array(centers, mask)
    to_donate = clusters[smallest].donation_candidates(centers)

    for _ in range(len(clusters[smallest]) - odd_size):
        pt, tar = next(to_donate)
        clusters[tar].add(tuple(pt))
        clusters[smallest].remove(tuple(pt))

    for it in range(max_iterations):
        # pdb.set_trace()
        by_size = np.argsort(np.array([c.count_donatable() for c in clusters]))

        print(it, by_size[-1], len(clusters[by_size[-1]]), sep="\t")

        largest = by_size[-1]
        mask = np.zeros(2 * len(centers), dtype=bool).reshape((-1, 2))
        mask[largest] = True
        to_donate = clusters[largest].donation_candidates(
                np.ma.masked_array(centers, mask))

        targeted = {largest}
        for _ in range(len(clusters[largest])
                - clusters[by_size[-2]].count_donatable() + 1):
            pt, tar = next(to_donate)
            clusters[largest].remove(tuple(pt))
            clusters[tar].add(tuple(pt))
            targeted.add(tar)

        for tar in targeted:
            centers[tar] = clusters[largest].all().mean(axis=0)

    return [c.all() for c in clusters]

def find_clusters(pixels, number_of_labels, max_per_cluster=100):
    w, h = pixels.shape

    sample_pop = np.ceil(np.bincount(np.reshape(pixels, -1)) / max_per_cluster).astype(int)

    colors = [[] for _ in range(len(sample_pop))]
    for x in range(w):
        for y in range(h):
            colors[pixels[x][y]].append((x, y))
    colors = [np.array(c) for c in colors]

    clusters = [None for _ in range(len(colors))]
    for c in range(len(colors)):
        # Run KMeans clustering
        clt = MiniBatchKMeans(n_clusters=sample_pop[c])
        clt.fit_predict(colors[c])
        centers, labels = clt.cluster_centers_, clt.labels_

        # Construct clusters
        clus = [set((x, y) for x, y in colors[c][labels == i]) for i in range(sample_pop[c])]
        clusters[c] = redistribute_clusters(clus, centers, max_per_cluster)

    return clusters


def solve_rounds(pixels, number_of_cans, max_pixels_per_can=100):
    return find_clusters(pixels, number_of_cans, max_per_cluster=max_pixels_per_can)
