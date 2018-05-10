#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

class MyLogRegressor():

    def __init__(self, kappa=0.01, max_iter=200):
        self._kappa = kappa
        self._max_iter = max_iter

    def fit(self, X, y):
        X = self.__feature_rescale(X)
        X = self.__feature_prepare(X)
        log_like = self.__batch_gradient_descent(X, y)
        return log_like

    def predict(self, X):
        ##############################
        #
        #  put your code here
        #
        ##############################
        pass

    
    def __batch_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        ll = []
        self._w = np.zeros(X.shape[1])
        ll.append(self.__log_like(X,y,self._w))
        ##############################
        #
        #  put your code here
        #
        ##############################
        return ll

    def __total_error(self, X, y, w):
        ##############################
        #
        #  put your code here
        #
        ##############################
        return 0

    def __log_like(self, X, y, w):
        ##############################
        #
        #  put your code here
        #
        ##############################
        return -100

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

    data = load_breast_cancer()
    X, y = data['data'], data['target']
    mylinreg = MyLogRegressor()
    print(mylinreg.fit(X, y))
