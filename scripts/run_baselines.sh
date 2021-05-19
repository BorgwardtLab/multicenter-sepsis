#!/bin/bash
result_path=$1
feature_set=middle
cost=5

methods=(sofa qsofa sirs mews news)
for method in ${methods[@]}; do
    for rep in {0..4}; do
        python -m src.sklearn.main --method $method --result_path $result_path --cv_n_jobs=15 --n_iter_search=50 --dataset=aumc --task classification --feature_set $feature_set --cost=$cost --rep $rep 
        python -m src.sklearn.main --method $method --result_path $result_path --cv_n_jobs=10 --n_iter_search=50 --dataset=mimic --task classification --feature_set $feature_set --cost=$cost --rep $rep
        python -m src.sklearn.main --method $method --result_path $result_path --cv_n_jobs=10 --n_iter_search=50 --dataset=hirid --task classification --feature_set $feature_set --cost=$cost --rep $rep
        python -m src.sklearn.main --method $method --result_path $result_path --cv_n_jobs=6 --n_iter_search=50 --dataset=eicu --task classification --feature_set $feature_set --cost=$cost --rep $rep
    done
done


