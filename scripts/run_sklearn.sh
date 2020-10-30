#!/bin/bash
result_path=$1
#LGBM
python -m src.sklearn.main --result_path $result_path --cv_n_jobs=20 --n_iter_search=100 --dataset=mimic3
python -m src.sklearn.main --result_path $result_path --cv_n_jobs=20 --n_iter_search=100 --dataset=physionet2019
python -m src.sklearn.main --result_path $result_path --cv_n_jobs=20 --n_iter_search=100 --dataset=hirid
python -m src.sklearn.main --result_path $result_path --cv_n_jobs=5 --n_iter_search=100 --dataset=eicu

#LR
python -m src.sklearn.main  --method lr --result_path $result_path --cv_n_jobs=20 --n_iter_search=100 --dataset=mimic3
python -m src.sklearn.main  --method lr --result_path $result_path --cv_n_jobs=5 --n_iter_search=100 --dataset=physionet2019
python -m src.sklearn.main  --method lr --result_path $result_path --cv_n_jobs=20 --n_iter_search=100 --dataset=hirid
python -m src.sklearn.main  --method=lr --result_path $result_path --cv_n_jobs=10  --n_iter_search=100 --dataset eicu  
