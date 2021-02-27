#!/bin/bash
#Demo first:

# first generate splits from raw data
# python -m src.splits.create_splits

# then run preprocessing pipeline per dataset

export DASK_DISTRIBUTED__COMM__RETRY__COUNT=20                                                                                                                                                                  
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="45s"
datasets=(eicu) #demo mimic

for dataset in ${datasets[@]}; do

    features=datasets/${dataset}/data/parquet/features
    split_file=config/splits/splits_${dataset}.json
 
    python -m src.preprocessing.extract_features datasets/downloads/${dataset}_0.3.0.parquet \
        --split-file $split_file \
        --output $features \
        --n-workers=30
   
    #for rep in {0..4}; do 
    #    normalizer_file=config/normalizer/normalizer_${dataset}_rep_${rep}.json
    #    lambda_file=config/lambdas/lambda_${dataset}_rep_${rep}.json
    #    python -m src.preprocessing.normalize 
    #        --input-file $features \
    #        --split-file $split_file \
    #        --split-name train \
    #        --repetition=$rep \
    #        --output-file $normalizer_file \
    #        --force
    #    python -m src.preprocessing.lambda \
    #        --input-file $features \
    #        --split-file $split_file \
    #        --split-name train \
    #        --repetition=$rep \
    #         --output-file config/lambdas/lambda_demo_rep_0.json
    #    done
    done

# previously:
# python -m src.sklearn.data.make_features --dataset demo --n_jobs=50 --n_partitions=20 --n_chunks=1
# python -m src.sklearn.data.make_features --dataset mimic --n_jobs=50 --n_partitions=20000 --n_chunks=1


