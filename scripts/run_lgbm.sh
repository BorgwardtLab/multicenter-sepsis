#!/bin/bash
result_path=$1
feature_set=middle
cost=5

#hp13:
#LGBM, result_path: results/hypersearch19_sklearn
#python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=aumc --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=mimic --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=hirid --task classification --feature_set $feature_set --cost=$cost
#python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=eicu --task classification --feature_set $feature_set --cost=$cost

#result_path: results/hypersearch19_physionet
# python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=physionet2019 --task classification --feature_set $feature_set --cost=$cost --variable_set=physionet
python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=aumc --task classification --feature_set $feature_set --cost=$cost --variable_set=physionet
python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=mimic --task classification --feature_set $feature_set --cost=$cost --variable_set=physionet
python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=hirid --task classification --feature_set $feature_set --cost=$cost --variable_set=physionet
python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=eicu --task classification --feature_set $feature_set --cost=$cost --variable_set=physionet

