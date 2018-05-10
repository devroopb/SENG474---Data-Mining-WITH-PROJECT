#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class MyLinearRegressor():

    def __init__(self, kappa=0.01, lamb=0, max_iter=500, opt='batch'):
        self._kappa = kappa
        self._lamb = lamb
        self._opt = opt
        self._max_iter = max_iter

    def fit(self, X, y):
        X = self.__feature_rescale(X)
        X = self.__feature_prepare(X)
        loss = []
        if self._opt == 'sgd':
            loss = self.__stochastic_gradient_descent(X, y)
        elif self._opt == 'batch':
            loss = self.__batch_gradient_descent(X, y)
        elif self._opt == 'adagrad':
            loss = self.__adaptive_gradient_descent(X, y)
        else:
            print('unknown opt')
        return loss

    def predict(self, X):
        pass

    def __batch_gradient_descent(self, X, y):
        M, N = X.shape
        niter = 1
        theta = np.ones(X.shape[1])
        loss = []
        while niter <= self._max_iter:
            hyp = np.dot(X, theta)
            theta -= self._kappa * np.dot(X.T, hyp - y)/M
            theta[1:] -= self._kappa * self._lamb/M
            loss.append(self.__total_loss(X, y, theta))
            niter += 1
        return loss

    def __stochastic_gradient_descent(self, X, y):
        M, N = X.shape
        niter = 1
        theta = np.ones(X.shape[1])
        loss = []
        while niter <= self._max_iter:
            idx = np.random.permutation(len(y))
            X, y = X[idx], y[idx]
            for i in range(M):
                hyp = np.dot(X[i, :], theta)
                theta -= self._kappa * (hyp - y[i]) * X[i, :]
                theta[1:] -= self._kappa * self._lamb/M
                loss.append(self.__total_loss(X, y, theta))
                niter += 1
        return loss

    def __total_loss(self, X, y, theta):
        tl = 0.5 * np.sum((np.dot(X, theta) - y)**2)/len(y)
        return tl

    def __feature_prepare(self, X_):
        M, N = X_.shape
        X = np.ones((M, N+1))
        X[:, 1:] = X_
        return X

    def __feature_rescale(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        return (X - self._mu)/self._sigma


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    import pylab as plt

    data = load_boston()
    X, y = data['data'], data['target']

    # batch
    mlinreg = MyLinearRegressor(kappa=0.001, opt='batch')
    l1 = mlinreg.fit(X, y)
    mlinreg = MyLinearRegressor(kappa=0.01, opt='batch')
    l2 = mlinreg.fit(X, y)
    mlinreg = MyLinearRegressor(kappa=0.1, opt='batch')
    l3 = mlinreg.fit(X, y)

    
    plt.clf()
    plt.plot(l1)
    plt.plot(l2)
    plt.plot(l3)
    plt.xlabel('#iter')
    plt.ylabel('E(w)')
    plt.legend(['k=.001','k=.01', 'k=.1'])
    plt.savefig('batch.pdf')

    # SGD
    mlinreg = MyLinearRegressor(kappa=0.0001, opt='sgd')
    l1 = mlinreg.fit(X, y)
    mlinreg = MyLinearRegressor(kappa=0.001, opt='sgd')
    l2 = mlinreg.fit(X, y)
    mlinreg = MyLinearRegressor(kappa=0.01, opt='sgd')
    l3 = mlinreg.fit(X, y)

    plt.clf()
    plt.plot(l1)
    plt.plot(l2)
    plt.plot(l3)
    plt.xlabel('#iter')
    plt.ylabel('E(w)')
    plt.legend(['k=.0001','k=.001', 'k=.01'])
    plt.savefig('sgd.pdf')


    # bonus question
    plt.clf()
    for lamb in [10, 100, 1000]:
        ml = MyLinearRegressor(kappa=0.01, lamb=lamb, opt='sgd')
        l = ml.fit(X, y)
        plt.plot(l)
    plt.xlabel("#training samples")
    plt.legend(['lambda=10', 'lambda=100', 'lambda=1000'])
    plt.ylabel("E(w)")
    plt.savefig('sgd-bonus.pdf')


