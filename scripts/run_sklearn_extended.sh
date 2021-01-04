#!/bin/bash
result_path=$1
##LGBM
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=20 --dataset=mimic3 --extended_features
python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=20 --dataset=physionet2019 --extended_features --feature_set=challenge
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=20 --dataset=hirid --extended_features
#python -m src.sklearn.main --result_path $result_path --cv_n_jobs=1 --n_iter_search=20 --dataset=eicu --extended_features

#LR
#python -m src.sklearn.main  --method lr --result_path $result_path --cv_n_jobs=1 --n_iter_search=20 --dataset=mimic3 --extended_features
#python -m src.sklearn.main  --method lr --result_path $result_path --cv_n_jobs=1 --n_iter_search=20 --dataset=physionet2019 --extended_features
#python -m src.sklearn.main  --method lr --result_path $result_path --cv_n_jobs=1 --n_iter_search=20 --dataset=hirid --extended_features
#python -m src.sklearn.main  --method=lr --result_path $result_path --cv_n_jobs=1  --n_iter_search=20 --dataset eicu --extended_features 
