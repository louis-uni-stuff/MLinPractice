""" Automized Hyperparameter Optimization for the MLP classifier """

#!/bin/bash

mkdir -p data/classification

# specify hyperparameter values
hidden_layer_sizes=("50 100 150 200")
activation=("identity logistic tanh relu")
solvers=("lbfgs sgd adam")
max_fun=("15000 22000 30000")

# different execution modes
if [ $1 = local ]
then
    echo "[local execution]"
    cmd="code/classification/classifier.sge"
elif [ $1 = grid ]
then
    echo "[grid execution]"
    cmd="qsub code/classification/classifier.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search
for size in $hidden_layer_sizes; do
for func in $activation; do
for solver in $solvers; do
for mf in $max_fun; do
    echo $size
    echo $func
    echo $solver
    echo $mf
    $cmd 'data/classification/clf_'"$size"'_'"$func"'_'"$solver"'_'"$mf"'.pickle' --seed 42 --mlp $size $func $solver $mf --accuracy --mcc --informedness --balanced_accuracy --kappa --f1_score
done
done
done
done