#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Solution courtesy of John Verwolf

import heapq
from Queue import PriorityQueue

import numpy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from numpy.random import randint
from collections import Counter


class MyBisectKmeans():

    def __init__(self, K):
        self._K = K

    def fit(self, X):
        # The first iteration is a simple KMeans with 1 centriod.
        clf = KMeans(1).fit(X)
        centriods = clf.cluster_centers_

        for _ in range(self._K - 1):
            # Get the index of the centriod for the max cluster.
            clf.cluster_centers_ = centriods
            max_index = self.largest_cluster(clf.predict(X))

            # Remove the centriod for the max cluster from the centriod list.
            centriods = np.delete(centriods, max_index, axis=0)

            # Get just the points in the max cluster.
            max_cluster = X[clf.predict(X) == max_index]

            # Bisect the max cluster. Use the best of 20 attempts.
            clf = KMeans(2, n_init=20).fit(max_cluster)

            # Add the new centriods to the centriod list.
            centriods = np.append(centriods, clf.cluster_centers_, axis=0)

        clf.cluster_centers_ = centriods
        return clf.predict(X)

    @classmethod
    def largest_cluster(cls, labels):
        counter = Counter(labels)
        return counter.most_common()[0][0]


# plot clusters and color them based on the cluster assignment in preds
def plot_clusters(data_in, preds):
    plt.clf()
    plt.scatter(data_in[:, 0], data_in[:, 1], c=preds)
    plt.axis('equal')
    plt.title("Cluster Assignments")
    plt.show()
    # May be of use for saving your plot:    plt.savefig(filename)


if __name__ == '__main__':
    # This is an easy way to make data sampled from clusters
    # with equal variance.  You can use the same method to change
    # the variance of individual clusters
    n_samples = 578
    X, y = make_blobs(n_samples=n_samples, centers=4)

    mbs = MyBisectKmeans(4)
    clusters_out = mbs.fit(X)
    plot_clusters(X, clusters_out)

    # Generate data with covarying dimensions
    # A linear algebra reminder of how to make transformation
    # matrices http://mathforum.org/mathimages/index.php/Transformation_Matrix
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, centers=4)
    transformation = [[1, 0], [1.5, 1]]
    X_shear = np.dot(X, transformation)

    clusters_out = mbs.fit(X_shear)
    plot_clusters(X_shear, clusters_out)
