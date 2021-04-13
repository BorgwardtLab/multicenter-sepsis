#!/bin/bash

# first generate splits from raw data
version=0.4.0
#python -m src.splits.create_splits --version $version 

# then run preprocessing pipeline per dataset

export DASK_DISTRIBUTED__COMM__RETRY__COUNT=30                                                                                                                                                                  
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="180s" #45s
datasets=(mimic_demo) #mimic aumc hirid eicu physionet2019) # aumc physionet2019 hirid eicu) #mimic_demo mimic 
cost=5
feature_sets=(small middle) #small
for dataset in ${datasets[@]}; do

    echo ">>> Processing $dataset ..."
    split_file=config/splits/splits_${dataset}_v${version}.json
    #for feature_set in ${feature_sets[@]}; do
    #    features=datasets/${dataset}/data/parquet/features_${feature_set}_v${version}
    #    rm -r $features 
    #    #echo ">>> Extracting features ..." 
    #    python -m src.preprocessing.extract_features datasets/downloads/${dataset}_${version}.parquet \
    #        --split-file $split_file \
    #        --output $features \
    #        --n-workers=20 \
    #        --feature_set=$feature_set 
    #done
    for rep in {0..4}; do # {0..4} 
        normalizer_file=config/normalizer/normalizer_${dataset}_rep_${rep}_v${version}.json
        
        lambda_file=config/lambdas/lambda_${dataset}_rep_${rep}_cost_${cost}_v${version}.json
        echo ">>> Split repetition $rep:"
        echo ">>> Normalization ..." 
        python -m src.preprocessing.normalize \
            --input-file $features \
            --split-file $split_file \
            --split-name train \
            --repetition=$rep \
            --output-file $normalizer_file \
            --force \
            --distributed
        echo ">>> Lambda calculation ..."
        python -m src.preprocessing.lambda \
            --input-file $features \
            --split-file $split_file \
            --split-name train \
            --repetition=$rep \
            --output-file $lambda_file \
            --n-jobs 50 \
            --cost $cost 
    done
done

# also extract all feature sets and write to json:
python -m src.variables.feature_groups --input_path $features 

# caching for deep models: (depends on feature groups)
splits=(train validation test)
for dataset in ${datasets[@]}; do
    for rep in {0..4}; do 
        for split in ${splits[@]}; do 
                    # caching for deep models:
                    python -m src.sklearn.loading \
                        --dataset $dataset \
                        --dump_name features_small_v${version} \
                        --cache_path datasets/${dataset}/data/parquet \
                        --split $split \
                        --rep $rep \
                        --cost $cost
        done
    done
done
