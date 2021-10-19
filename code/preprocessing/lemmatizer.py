#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that reduces words to their lemmas.
@author: louiskhub
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import ast


class Lemmatizer(Preprocessor):
    """Reduces words to their lemmas."""

    def __init__(self, input_col, output_col):
        """Initialize the Lemmatizer with the given input and output column."""
        super().__init__([input_col], output_col)
    
    # no need to implement _set_variables
    
    def _get_values(self, inputs):
        """Lemmatize the words."""
        
        lemmatized_col = []                             # We later append the lemmatized tweets to this list
        lemmatizer = WordNetLemmatizer()
        for row in inputs[0]:
            lemmatized = []
            iterable = ast.literal_eval(row)            # Make the string returned by the Tokenizer iterable
            for token, tag in pos_tag(iterable):
                
                pos = tag[0].lower()
                if pos not in ['a' , 'r' , 'n' , 'v']:  # Check the token's Part-of-Speech Tag for better lemmatization
                    pos = 'n'                           # Default is 'Noun'
        
                lemma = lemmatizer.lemmatize(token,pos) # Create lemmas with the Wordnet Lemmatizer
                lemmatized.append(lemma)
            lemmatized_col.append(lemmatized)
        return lemmatized_col
