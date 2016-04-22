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

def redistribute_clusters(clusters, centers, cluster_size):
    odd_size = sum([len(c) for c in clusters]) % cluster_size

    by_size = np.argsort(np.array([len(c) for c in clusters]))

    def donation_candidates(from_idx, centers, count=1):
        clus = np.array(list(clusters[from_idx]))
        tree = spatial.KDTree(centers)
        dists, target = tree.query(clus)
        by_dist = np.argsort(dists)
        return zip(clus[by_dist], target[by_dist])

    # Form the smallest cluster by giving away pixels from it
    mask = np.zeros(2 * len(centers), dtype=bool).reshape((-1, 2))
    smallest = by_size[0]
    mask[smallest] = True
    centers = np.ma.masked_array(centers, mask)
    to_donate = donation_candidates(smallest, centers)
    for _ in range(len(clusters[smallest]) - odd_size):
        pt, tar = next(to_donate)
        clusters[tar].add(tuple(pt))
        clusters[smallest].remove(tuple(pt))

    # TODO: once a pixel has been donated, never donate it again.
    # But we have to be very careful not to donate a pixel to a cluster which
    # already has 100 donated pixels.
    it = 0
    mask = np.zeros(2 * len(centers), dtype=bool).reshape((-1, 2))
    mask[smallest] = True
    while not all(len(c) <= cluster_size for c in clusters):
        by_size = np.argsort(np.array([len(c) for c in clusters]))

        print(it, by_size[-1], len(clusters[by_size[-1]]), sep="\t")
        it += 1

        largest = by_size[-1]
        # if mask[largest].all():
        #     mask = np.zeros(2 * len(centers), dtype=bool).reshape((-1, 2))
        mask[largest] = True
        if mask.all():
            mask = np.zeros(2 * len(centers), dtype=bool).reshape((-1, 2))
            mask[smallest], mask[largest] = True, True

        to_donate = donation_candidates(largest,
                np.ma.masked_array(centers, mask).
                filled((float("inf"), float("inf"))))

        for _ in range(len(clusters[largest]) - len(clusters[by_size[-2]]) + 2):
            pt, tar = next(to_donate)
            clusters[largest].remove(tuple(pt))
            clusters[tar].add(tuple(pt))

        centers[largest] = np.array(list(clusters[largest])).mean(axis=0)

    return clusters

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
