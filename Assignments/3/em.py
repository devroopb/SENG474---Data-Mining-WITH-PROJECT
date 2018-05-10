#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# good for calculating the probability of a point
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
from numpy.random import randint

# Please implement the EM algorithm for the case where the variance
# of each cluster is a diagonal matrix, but the variance of each
# dimension can be different.
# You may find this scipy library to be of use
# https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.multivariate_normal.html
class MyEM():

    def __init__(self, K):
        self._K = K

    def fit(self, X):
        # your code goes here
        # you should return an array of ints corresponding to the cluster 
        # assignments [0 ... K-1]
        # Rescale the data so that they won't be too far from
        # your randomly chosen normal distributions 
        X = self.__feature_rescale(X)
        return randint(0,self._K,len(X))

    # rescale dimensions to mean=0, std=1
    # note that this does not use any information
    # about the clustering assignment, and treats
    # all data points equally
    def __feature_rescale(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        return (X - self._mu)/self._sigma


# plot clusters and color them based on the cluster assignment in preds
def plot_clusters(data_in, preds):
    plt.clf()
    plt.scatter(data_in[:, 0], data_in[:, 1], c=preds)
    plt.axis("equal")
    plt.title("Cluster Assignments")
    plt.show()
    # May be of use for saving your plot:    plt.savefig(filename)


if __name__ == '__main__':
    # This is an easy way to make data sampled from clusters
    # with equal variance.  You can use the same method to change
    # the variance of individual clusters
    n_samples = 100
    X, y = make_blobs(n_samples=n_samples, centers = 3)

    mem = MyEM(3)
    clusters_out = mem.fit(X)
    plot_clusters(X, clusters_out)

    # Generate data where the dimension have different variance
    # A linear algebra reminder of how to make transformation
    # matrices http://mathforum.org/mathimages/index.php/Transformation_Matrix
    X, y = make_blobs(n_samples=n_samples, centers = 3)
    X_shear = X
    # this is a scaling transformation (different variance in diff dims)
    # we will apply a different transformation to each of the 3 clusters
    transformation = [[5, 0], [0, 1]]
    X_shear[y==0] = np.dot(X[y==0], transformation)
    transformation = [[1, 0], [0, 3]]
    X_shear[y==1] = np.dot(X[y==1], transformation)
    transformation = [[10, 0], [0, 1]]
    X_shear[y==2] = np.dot(X[y==2], transformation)

    clusters_out = mem.fit(X_shear)
    plot_clusters(X_shear, clusters_out)





