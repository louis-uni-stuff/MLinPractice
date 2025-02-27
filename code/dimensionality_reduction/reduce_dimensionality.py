#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a dimensionality reduction technique.
@author: lbechberger
"""

import argparse, pickle
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


# setting up CLI
parser = argparse.ArgumentParser(description = "Dimensionality reduction")
# mandatory
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("output_file", help = "path to the output pickle file")
# optional
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("--verbose", action = "store_true", help = "print information about feature selection process")
# dimensionality reducers
parser.add_argument("-m", "--mutual_information", type = int, help = "select K best features with Mutual Information", default = None)
parser.add_argument("--tsvd", action = "store_true", help = "find embedding space using truncated SVD", default = None)
parser.add_argument("-p", "--pca", type = int, help = "projects features into K main dimensions of variation using 'Principle Component Analysis'", default = None)

args = parser.parse_args()

# load the data
with open(args.input_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

features = input_data["features"]
labels = input_data["labels"]
feature_names = input_data["feature_names"]

if args.import_file is not None:
    # simply import an already fitted dimensionality reducer
    with open(args.import_file, 'rb') as f_in:
        dim_red = pickle.load(f_in)

else: # need to set things up manually

    if args.mutual_information is not None:
        # select K best based on Mutual Information
        dim_red = SelectKBest(mutual_info_classif, k = args.mutual_information)
        dim_red.fit(features, labels.ravel())
        
        # resulting feature names based on support given by SelectKBest
        def get_feature_names(kbest, names):
            support = kbest.get_support()
            result = []
            for name, selected in zip(names, support):
                if selected:
                    result.append(name)
            return result
        
        if args.verbose:
            print("    SelectKBest with Mutual Information and k = {0}".format(args.mutual_information))
            print("    {0}".format(feature_names))
            print("    " + str(dim_red.scores_))
            print("    " + str(get_feature_names(dim_red, feature_names)))

    elif args.tsvd:
        dim_red = TruncatedSVD(n_components=25, algorithm='arpack')
        dim_red.fit(features, labels.ravel())

    elif args.pca is not None:
        
        pca = PCA(n_components=args.pca)
        dim_red = pca.fit(features)

        if args.verbose:
            print("\n")
            print("    Projecting features into {0} main dimensions of variation using PCA".format(args.pca))
            print ("    Composition of the {0} dimensions:".format(args.pca))
            print("\n")
            for component in pca.components_:
                print (" + ".join("    %.2f    x    %s    " % (value, name) for value, name in zip(component, feature_names)))
                print("\n")
    pass

# apply the dimensionality reduction to the given features
reduced_features = dim_red.transform(features)

# store the results
output_data = {"features": reduced_features, 
               "labels": labels}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(output_data, f_out)

# export the dimensionality reduction technique as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(dim_red, f_out)