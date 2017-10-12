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
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = [] # probability of class given feature
        self._class_prob = [] # probability of each class
        self._Ncls = []
        self._Nfeat = []

    def train(self, X, y):        
        # calculate P of class
        unique, counts = np.unique(y, return_counts = True) # unique is [0,1,2,3], counts is [480,584,593,377]
        self._Ncls = len(unique) # self._Ncls is 4 classes
        self._Nfeat = X.shape[0]
        self._class_prob = counts/len(y) # len(y) is 2034, total number of non-unique classes
		
        # calculate P of feature given class
        # create new array of fixed size equal to number of classes
        spliton = [None] * (len(unique) - 1)
        # determine where the data get split up by each unique type
        spliton[0] = counts[0]
        for i in range(1, len(unique) - 1):
            spliton[i] = counts[i] + spliton[i - 1]
        self._Nfeat = len(X[0]) #self._Nfeat is 26576
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
        self._feat_prob = np.true_divide(np.add(cls_arrs, self._smooth), np.sum(cls_arrs) + self._Nfeat * self._smooth)

    def predict(self, X):
        

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

