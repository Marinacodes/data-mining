#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(666)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_boundary(clf, X_train, Y_train, xx, yy):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train,
                cmap=plt.cm.coolwarm,
                edgecolors='k')
    plt.show()


class Perceptron():

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = np.array([[5., 0., 5.]])

    def fit_epoch(self, X, Y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        shuff = np.random.permutation(len(Y))
        X, Y = X[shuff], Y[shuff]
        #########################################
        # insert your code here
        #########################################

    def predict(self, X):
        asd = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.sign(np.dot(asd, self.weights.T))


if __name__ == '__main__':
    N, M = 40, 2
    X_train = np.r_[np.random.randn(N, M) + [1, 1], np.random.randn(N, M) + [10, 10]]
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    Y_train = np.array([1]*N + [-1]*N)

    xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])

    ########################################
    # insert your code here
    #########################################
    n_epochs = 100
    clf = Perceptron(learning_rate=0.05)
