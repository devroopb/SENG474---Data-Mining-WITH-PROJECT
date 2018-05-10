#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from numpy.random import randint
from numpy.random import randn

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
        X = self.__feature_rescale(X)
        mus = randn(self._K, X.shape[1])
        sigmas = np.zeros([self._K, X.shape[1], X.shape[1]])
        for i in range(self._K):
            sigmas[i,:,:] = np.eye(X.shape[1])
        ws = np.ones(self._K)*(1.0/len(X))

        ps = self._e_step(X, mus, sigmas, ws)
        [mus, sigmas, ws ] = self._m_step(X, ps, mus, sigmas, ws)
        diff = 100
        while diff > 0.001:
            labs = np.argsort(ps,axis=1)[:,-1]
            tmp_labs = np.append(labs, np.ones(self._K)*self._K)
            tmp_data = np.append(X, mus,axis=0)
            # if i%10 == 0:
            #     plot_clusters(tmp_data, tmp_labs)
            ps = self._e_step(X, mus, sigmas, ws)
            [new_mus, sigmas, ws ] = self._m_step(X, ps, mus, sigmas, ws)
            diff = np.mean(np.abs(new_mus-mus))
            mus = new_mus
            print diff
        labs = np.argsort(ps,axis=1)[:,-1]
        tmp_labs = np.append(labs, np.ones(self._K)*self._K)
        tmp_data = np.append(X, mus,axis=0)
        # plot_clusters(tmp_data, tmp_labs)
        return labs

    def _e_step(self, X, mus, sigmas, ws):
        raw_ps = np.zeros([len(X),self._K])
        for i in range(self._K):
            rv = multivariate_normal(mus[i], sigmas[i,:,:])
            raw_ps[:,i] = rv.pdf(X)
        print raw_ps
        print ws
        ps = np.multiply(ws,raw_ps)
        ps = np.divide(ps.T,np.sum(ps,axis=1)).T

        return ps

    def _m_step(self, X, ps, mus, sigmas, ws):
        new_mus = np.zeros(mus.shape)
        for i in range(self._K):
            tmp = np.multiply(X.T,ps[:,i]).T
            tmp = np.divide(np.mean(tmp,axis=0),np.mean(ps[:,i]))
            new_mus[i,:] = tmp

        print new_mus
        print "\n"

        new_sigmas = np.zeros(sigmas.shape)
        for i in range(self._K):
            diff_sq = np.power(np.subtract(X,mus[i,:]),2);
            tmp = np.multiply(diff_sq.T,ps[:,i]).T
            tmp = np.divide(np.mean(tmp,axis=0),np.mean(ps[:,i]))
            new_sigmas[i,:,:] = np.multiply(np.eye(X.shape[1]),tmp)
        new_ws = np.mean(ps,axis=0)
        print new_ws
        print "\n"
        return [new_mus, new_sigmas, new_ws ]

    # rescale features to mean=0 and std=1
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
    n_samples = 200
    X, y = make_blobs(n_samples=n_samples, centers = [[0,0],[0,2],[0,4]],cluster_std=[1,0.1,1])


    mem = MyEM(3)
    kmeans = KMeans(n_clusters=3);
    clusters_out = mem.fit(X)
    clusters_km = kmeans.fit(X)
    plot_clusters(X, y)
    plot_clusters(X, clusters_out)
    plot_clusters(X,clusters_km.labels_)

    # Generate data where the dimension have different variance
    # A linear algebra reminder of how to make transformation
    # matrices http://mathforum.org/mathimages/index.php/Transformation_Matrix
    X, y = make_blobs(n_samples=n_samples, centers = [[0,0],[0,1],[0,2]])
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
    clusters_km = kmeans.fit(X)

    plot_clusters(X_shear, y)
    plot_clusters(X_shear, clusters_out)
    plot_clusters(X_shear, clusters_km.labels_)






