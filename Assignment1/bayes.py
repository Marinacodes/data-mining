#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time

#np.set_printoptions(threshold=np.nan)

class MyBayesClassifier():
    # For graduate and undergraduate students to implement Bernoulli Bayes
    def __init__(self, smooth = 1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []
		
    def calculate_P_of_class(self, y):
        unique, counts = np.unique(y, return_counts = True) # unique is [0,1,2,3], counts is [480,584,593,377]
        self._Ncls = len(unique) # self._Ncls is 4 classes
        self._class_prob = counts/len(y) # len(y) is 2034, total number of non-unique classes
        #print self._class_prob
        return self._class_prob
		
    def calculate_P_of_feature_given_class(self, X, y):
        unique, counts = np.unique(y, return_counts = True) # unique is [0,1,2,3], counts is [480,584,593,377]
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
        # print class arrays
        for i in range(0, len(unique)):
            print cls_arrs[i]
        print self._feat_prob
        return self._feat_prob

    def train(self, X, y):
        # Your code goes here.
        self.calculate_P_of_class(y)	
        self.calculate_P_of_feature_given_class(X, y)
        return

    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        return np.zeros([X.shape[0],1])

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

