#!/bin/bash
result_path=$1
feature_set=middle
variable_set=physionet #full
cost=5

#hp13
## LGBM REPETITIONS:
datasets=(mimic hirid eicu) #physionet2019 aumc)
for dataset in ${datasets[@]}; do
    for rep in {0..4}; do
        python -m src.sklearn.fit_rep --method lgbm --result_path $result_path --dataset=$dataset --task classification --feature_set $feature_set --cost=$cost --rep $rep --variable_set $variable_set
    done 
done
