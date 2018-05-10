#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math


class MyLogRegressor():

    def __init__(self, kappa=0.01, max_iter=500):
        self._kappa = kappa
        self._max_iter = max_iter

    def fit(self, X, y):
        X = self.__feature_rescale(X)
        X = self.__feature_prepare(X)
        log_like = self.__batch_gradient_descent(X, y)
        return log_like

    def predict(self, X):
        return np.exp(-1*np.dot(X,self._w))<1

    
    def __batch_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        ll = []
        self._w = np.zeros(X.shape[1])
        for niter in range(self._max_iter):
            X_w = np.dot(X,self._w)
            tile_prod = np.tile(y-(np.exp(X_w)/(1+np.exp(X_w))),(31,1)).T
            prod = y-(np.exp(X_w)/(1+np.exp(X_w)))
            self._w = self._w + self._kappa*np.sum((X.T*prod),1)
            ll.append(self.__log_like(X,y,self._w))
        return ll

    def __total_error(self, X, y, w):
        tl = 1-np.mean(self.predict(X)==y)
        return tl

    def __log_like(self, X, y, w):
        X_w = np.dot(X,self._w)
        ll = np.sum(np.dot(y,X_w)-np.sum(np.log(1+np.exp(X_w))))
        return ll

    # add a column of 1s to X
    def __feature_prepare(self, X_):
        M, N = X_.shape
        X = np.ones((M, N+1))
        X[:, 1:] = X_
        return X

    # rescale features to mean=0 and std=1
    def __feature_rescale(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        return (X - self._mu)/self._sigma


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    import pylab as plt

    data = load_breast_cancer()
    X, y = data['data'], data['target']

    plt.clf()
    lgnd = [];
    for k in [0.00001, 0.0001, 0.001, 0.01]:
        ml = MyLogRegressor(kappa=k)
        l = ml.fit(X, y)
        plt.plot(l)
        lgnd.append('k='+str(k))

    plt.xlabel('#iter')
    plt.ylabel('E(w)')
    plt.legend(lgnd)
    plt.savefig('log-batch.pdf')


