#!/bin/bash
#Demo first:
python src/sklearn/data/make_dataframe.py --dataset demo --n_jobs=10 --n_partitions=20 --overwrite
python src/sklearn/data/create_instance_files.py --dataset demo

#Process the full datasets:
python src/sklearn/data/make_dataframe.py --dataset mimic3 --n_jobs=50 --n_partitions=20000 --overwrite
python src/sklearn/data/make_dataframe.py --dataset hirid --n_jobs=50 --n_partitions=20000 --overwrite
python src/sklearn/data/make_dataframe.py --dataset physionet2019 --n_jobs=50 --n_partitions=10000 --overwrite
python src/sklearn/data/make_dataframe.py --dataset eicu --n_jobs=50 --n_partitions=40000 --overwrite

python src/sklearn/data/create_instance_files.py --dataset mimic3
python src/sklearn/data/create_instance_files.py --dataset hirid
python src/sklearn/data/create_instance_files.py --dataset physionet2019
python src/sklearn/data/create_instance_files.py --dataset eicu

# Sanity checks: 
python scripts/check_splits.py
