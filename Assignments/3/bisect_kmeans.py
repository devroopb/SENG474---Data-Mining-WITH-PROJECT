#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from numpy.random import randint

class MyBisectKmeans():

    def __init__(self, K):
        self._K = K

    def fit(self, X):
        # your code goes here
        # you are welcome to use scikit learn's implementation of kmeans
        # but you must implement the bisecting algorithm yourself
        # you should return an array of ints corresponding to the cluster 
        # assignments [0 ... K-1]
        # This is stand in code that just randomly assigns points to clusters
        return randint(0,self._K,len(X))


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
    X, y = make_blobs(n_samples=n_samples, centers = 4)

    mbs = MyBisectKmeans(4)
    clusters_out = mbs.fit(X)
    plot_clusters(X, clusters_out)

    # Generate data with covarying dimensions
    # A linear algebra reminder of how to make transformation
    # matrices http://mathforum.org/mathimages/index.php/Transformation_Matrix
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, centers = 4)
    transformation = [[1, 0], [1.5, 1]]
    X_shear = np.dot(X, transformation)

    clusters_out = mbs.fit(X_shear)
    plot_clusters(X_shear, clusters_out)





