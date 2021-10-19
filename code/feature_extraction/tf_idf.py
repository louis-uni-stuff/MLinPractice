#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature that computes the term frequency–inverse document frequency in the given column.
@author: marcelklehr
"""
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer

class TfIdf(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_tfidf".format(input_column))
        self._vectorizer = TfidfVectorizer(input='content', max_features=200)

    # compute IDFs
    def _set_variables(self, inputs):
        return self._vectorizer.fit(inputs[0])

    # compute the tf-idf matrix
    def _get_values(self, inputs):
        print('TF-IDF vocabulary: {0}'.format(self._vectorizer.get_feature_names()))
        return self._vectorizer.transform(inputs[0]).toarray()