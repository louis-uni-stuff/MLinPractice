#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.
@author: lbechberger
"""

import argparse, pickle
from numpy import true_divide
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from mlflow import log_metric, log_param, set_tracking_uri


# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
# mandatory
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("--log_dir", help = "where to log the mlflow results", default = "data/classification/mlflow")
# optional
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
# baseline classifiers
parser.add_argument("-q", "--frequency", action = "store_true", help = "label-frequency class classifier", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier", default = None)
# classifiers
parser.add_argument("--knn", action = "store_true", help = "use KNN classifier", default=None)
parser.add_argument("--svm", action = "store_true", help = "svm classifier", default = None)
parser.add_argument("--mlp", nargs = "*", action = "append", help = "Multi-Layered-Perceptron classifier " 
                                                                    "<column> hyperparam1 hyperparam2 ... "
                                                                    "Available hyperparams in the correct order of application: "
                                                                    "hidden_layer_sizes, activation, solver, max_fun ",
                                                                    default = None)
# evaluation metrics
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
parser.add_argument("-b", "--balanced_accuracy", action = "store_true", help = "evaluate using balanced accuracy")
parser.add_argument("-f", "--f1_score", action = "store_true", help = "evaluate using the F1 score (or F-measure)")
parser.add_argument("-n", "--informedness", action = "store_true", help = "evaluate using informedness")
parser.add_argument("-k", "--kappa", action = "store_true", help = "evaluate using Cohen's kappa")
parser.add_argument("--mcc", action = "store_true", help = "evaluate using Mathews Correlation coefficient")

args = parser.parse_args()


# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

set_tracking_uri(args.log_dir)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        input_dict = pickle.load(f_in)

    classifier = input_dict["classifier"]
    
    # Logs for MLflow
    for param, value in input_dict["params"].items():
        log_param(param, value)
    log_param("dataset","validation")

else:   # manually set up a classifier
    
    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")

        log_param("classifier", "majority") # Log for MLflow
        params = {"classifier": "majority"}

        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
        classifier.fit(data["features"], data["labels"])

    elif args.frequency:
        # label-frequency classifier
        print("    label-frequency classifier")
        
        log_param("classifier", "frequency") # Log for MLflow
        params = {"classifier": "frequency"}

        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
        classifier.fit(data["features"], data["labels"])

    elif args.svm:
        #  Support Vector Machine
        print("    SVM classifier")

        log_param("classifier", "svm") # Log for MLflow
        params = {"classifier": "svm"}

        classifier = LinearSVC(dual=False, class_weight='balanced', random_state = args.seed)
        classifier.fit(data["features"], data["labels"].ravel())

    elif args.knn:
        # KNN classifier
        print("    KNN classifier")

        log_param("classifier", "knn") # Log for MLflow
        params = {"classifier": "knn"}

        classifier = KNeighborsClassifier(algorithm="auto", weights="distance", n_neighbors=10, random_state = args.seed)
        classifier.fit(data["features"], data["labels"].ravel())

    elif args.mlp:
        #MLP classifier
        print("    MLP classifier")
        
        # Fill hyperparams with default values if not specified in CLI
        args.mlp = [(100,), "identity", "adam", 15000] if args.mlp == [[]] else [item for sublist in args.mlp for item in sublist]
        
        # Logs for MLflow
        log_param("classifier", "mlp")
        log_param("hidden_layer_sizes", args.mlp[0])
        log_param("activation", args.mlp[1])
        log_param("solver", args.mlp[2])
        log_param("max_fun", args.mlp[3])
        params = {"classifier": "mlp", "hidden_layer_sizes": args.mlp[0], "activation": args.mlp[1], "solver": args.mlp[2], "max_fun": args.mlp[3]}

        classifier = MLPClassifier(hidden_layer_sizes = (int(args.mlp[0]),), activation = args.mlp[1], solver = args.mlp[2], max_fun = int(args.mlp[3]), random_state = args.seed)
        classifier.fit(data["features"], data["labels"].ravel())


# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("Accuracy", accuracy_score))
if args.balanced_accuracy:
    evaluation_metrics.append(("Balanced accuracy", balanced_accuracy_score))
if args.informedness:
    evaluation_metrics.append(("Informedness", lambda x,y: balanced_accuracy_score(x,y, adjusted=True)))
if args.kappa:
    evaluation_metrics.append(("Cohens kappa score", cohen_kappa_score))
if args.f1_score:
    evaluation_metrics.append(("F1 score", f1_score))
if args.mcc:
    # Division by zero may happen in this function, which produces a warning
    # see https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    evaluation_metrics.append(("MCC", matthews_corrcoef))

# compute and print them
for metric_name, metric in evaluation_metrics:
    metric_value = metric(data["labels"], prediction)
    print("    {0}: {1}".format(metric_name, metric_value))

    log_metric(metric_name, metric_value) # Log for MLflow
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    output_dict = {"classifier": classifier, "params": params}
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(output_dict, f_out)
        
        