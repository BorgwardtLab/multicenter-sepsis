#!/bin/bash
result_path=$1
feature_set=middle
variable_set=physionet #full
cost=5

#hp10:
## LGBM REPETITIONS:
datasets=(aumc mimic hirid physionet2019)
for dataset in ${datasets[@]}; do
    for rep in {0..4}; do
        python -m src.sklearn.fit_rep --method lr --result_path $result_path --dataset=$dataset --task classification --feature_set $feature_set --cost=$cost --rep $rep --variable_set $variable_set &
    done 
done

wait

for rep in {0..4}; do
    python -m src.sklearn.fit_rep --method lr --result_path $result_path --dataset=eicu --task classification --feature_set $feature_set --cost=$cost --rep $rep --variable_set $variable_set &
done 

