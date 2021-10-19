#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that checks whether this is a tweet as part of a thread
@author: marcelklehr
"""

import numpy as np
import re
from code.feature_extraction.feature_extractor import FeatureExtractor

class Threads(FeatureExtractor):
    """class for extracting the character-based length as a feature"""
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_threads".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):
        result = np.array(inputs[0].str.contains(r'🧵|^\s?[0-9]/[0-9]?\s|\s[0-9]/[0-9]?\s?$', flags=re.IGNORECASE, regex=True))
        result = result.reshape(-1, 1)
        return result
