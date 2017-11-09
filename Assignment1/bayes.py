#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time

class MyBayesClassifier():
    def __init__(self, smooth = 1):
        self._smooth = smooth # add one smoothing
        self._feat_prob = [] # probability of class given feature
        self._class_prob = [] # probability of each class
        self._Ncls = [] # number of classes
        self._Nfeat = [] # number of features

    def train(self, X, y):        
        # calculate P of class
        unique, counts = np.unique(y, return_counts = True) 
        self._Ncls = len(unique) 
        self._Nfeat = X.shape[0]
        self._class_prob = counts/len(y)
		
        # calculate P of feature given class
        # create new array of fixed size equal to number of classes
        spliton = [None] * (len(unique) - 1)
        # determine where the data get split up by each unique type
        spliton[0] = counts[0]
        for i in range(1, len(unique) - 1):
            spliton[i] = counts[i] + spliton[i - 1]
        self._Nfeat = len(X[0]) 
        # orient class array verically
        vert_y = np.vstack(y)
        # append class array to sample/feature array
        concat_arr = np.hstack((X, vert_y))
        # sort concatenated array by last column
        concat_arr = concat_arr[np.argsort(concat_arr[:, len(concat_arr[0]) - 1])]
        # split concatenated array by class
        cls_arrs = np.split(concat_arr, spliton, axis = 0)
        # delete the classes column (last one) and total feature columns over all samples
        for i in range(0, len(unique)):
            cls_arrs[i] = np.delete(cls_arrs[i], np.s_[-1], axis = 1)
            cls_arrs[i] = cls_arrs[i].sum(axis = 0)
        # calculate the probability that a feature belongs to a class
        self._feat_prob = np.true_divide(np.add(cls_arrs, self._smooth), np.sum(cls_arrs) + self._Nfeat * self._smooth)

    def predict(self, X):
        # create new array of fixed size equal to number of classes to store predictions in
        pred = [None] * len(X)
        # determine class 
        for row_num, row_contents in enumerate(X):
            pred[row_num] = self.determine_class(row_contents, self._feat_prob)
        return pred

    def determine_class(self, row_contents, feat_prob):
        # create new feature probability array
        feat_prob_not = np.empty((self._Ncls, self._Nfeat))
        # copy feature probability array into a new feature probability array
        feat_prob_not = np.copy(self._feat_prob)
        # create array of probabilities
        p = []
        # calculate probabilities
        for i in range(0, self._Ncls):
            feat_prob_not[i][row_contents == 0] = 1 - feat_prob[i][row_contents == 0]
            p.append(np.add(np.log(self._class_prob[i]), np.sum(np.log(feat_prob_not[i]))))
        # find the highest probability
        return np.argmax(p)

""" 
Here is the calling code

"""

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a count vectorizer")
t0 = time()

vectorizer = CountVectorizer(stop_words='english', binary=True)

X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train, y_train)
y_pred = clf.predict(X_test)
print 'alpha=%i accuracy = %f' %(alpha, np.mean((y_test-y_pred) == 0))