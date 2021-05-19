#!/bin/bash
result_path=$1
feature_set=middle
cost=5

#hp13:
#LGBM:
python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=aumc --task classification --feature_set $feature_set --cost=$cost
python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=mimic --task classification --feature_set $feature_set --cost=$cost
python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=hirid --task classification --feature_set $feature_set --cost=$cost
python -m src.sklearn.main --method lgbm  --result_path $result_path --cv_n_jobs=1 --n_iter_search=50 --dataset=eicu --task classification --feature_set $feature_set --cost=$cost

