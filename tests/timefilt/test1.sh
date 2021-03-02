#!/bin/bash

# first generate splits from raw data
# python -m src.splits.create_splits

# then run preprocessing pipeline per dataset

export DASK_DISTRIBUTED__COMM__RETRY__COUNT=20                                                                                                                                                                  
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="45s"
datasets=(demo)  

for dataset in ${datasets[@]}; do

    echo ">>> Processing $dataset ..."
    features=tests/timefilt/features
    split_file=config/splits/splits_${dataset}.json
    rm -r $features 
    echo ">>> Extracting features ..." 
    python -m tests.timefilt.dask_pipe tests/timefilt/data/${dataset}.parquet \
        --split-file $split_file \
        --output $features \
        --n-workers=30
   
    #for rep in {0..4}; do 
    #    normalizer_file=config/normalizer/normalizer_${dataset}_rep_${rep}.json
    #    lambda_file=config/lambdas/lambda_${dataset}_rep_${rep}.json
    #    echo ">>> Split repetition $rep:"
    #    echo ">>> Normalization ..." 
    #    python -m src.preprocessing.normalize \
    #        --input-file $features \
    #        --split-file $split_file \
    #        --split-name train \
    #        --repetition=$rep \
    #        --output-file $normalizer_file \
    #        --force
    #    echo ">>> Lambda calculation ..."
    #    python -m src.preprocessing.lambda \
    #        --input-file $features \
    #        --split-file $split_file \
    #        --split-name train \
    #        --repetition=$rep \
    #        --output-file $lambda_file 
    #    done
    done

# previously:
# python -m src.sklearn.data.make_features --dataset demo --n_jobs=50 --n_partitions=20 --n_chunks=1
# python -m src.sklearn.data.make_features --dataset mimic --n_jobs=50 --n_partitions=20000 --n_chunks=1

