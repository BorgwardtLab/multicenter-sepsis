#!/bin/bash

input_dir=results/hypersearch19_sklearn
methods=(lgbm sofa qsofa sirs mews news)

for method in ${methods[@]}; do
    source scripts/eval_sklearn.sh $input_dir $method &
done

