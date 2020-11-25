#!/bin/bash
#Demo first:
python -m src.sklearn.data.make_dataframe --dataset demo --n_jobs=10 --n_partitions=20 --overwrite
python src/sklearn/data/create_instance_files.py --dataset demo

#Process the full datasets:
python -m src.sklearn.data.make_dataframe --dataset physionet2019 --n_jobs=50 --n_partitions=10000 --overwrite
python src/sklearn/data/create_instance_files.py --dataset physionet2019

python -m src.sklearn.data.make_dataframe --dataset mimic3 --n_jobs=50 --n_partitions=20000 --overwrite
python src/sklearn/data/create_instance_files.py --dataset mimic3

python -m src.sklearn.data.make_dataframe --dataset hirid --n_jobs=50 --n_partitions=20000 --overwrite
python src/sklearn/data/create_instance_files.py --dataset hirid

python -m src.sklearn.data.make_dataframe --dataset eicu --n_jobs=50 --n_partitions=40000 --overwrite
python src/sklearn/data/create_instance_files.py --dataset eicu

python -m src.sklearn.data.make_dataframe --dataset aumc --n_jobs=50 --n_partitions=10000 --overwrite
python src/sklearn/data/create_instance_files.py --dataset aumc 

# Sanity checks: 
python scripts/check_splits.py
