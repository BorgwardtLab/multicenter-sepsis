#!/bin/bash
result_path=$1
feature_set=middle
cost=5

#hp13
## LGBM REPETITIONS:
for rep in {0..4}; do
    python -m src.sklearn.fit_rep --method lgbm --result_path $result_path --dataset=aumc --task classification --feature_set $feature_set --cost=$cost --rep $rep
    python -m src.sklearn.fit_rep --method lgbm --result_path $result_path --dataset=mimic --task classification --feature_set $feature_set --cost=$cost --rep $rep
    python -m src.sklearn.fit_rep --method lgbm --result_path $result_path --dataset=hirid --task classification --feature_set $feature_set --cost=$cost --rep $rep
    python -m src.sklearn.fit_rep --method lgbm --result_path $result_path --dataset=eicu --task classification --feature_set $feature_set --cost=$cost --rep $rep
done 

