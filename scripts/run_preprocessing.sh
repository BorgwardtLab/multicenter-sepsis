#!/bin/bash
#Demo first:

# first generate splits from raw data
python -m src.splits.create_splits

# then run preprocessing pipeline per dataset
python -m src.sklearn.data.make_features --dataset demo --n_jobs=50 --n_partitions=20 --n_chunks=1
python -m src.sklearn.data.make_features --dataset mimic --n_jobs=50 --n_partitions=20000 --n_chunks=1


