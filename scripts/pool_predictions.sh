#!/bin/bash
datasets=(hirid eicu aumc mimic)
output_path=results/evaluation_test/prediction_pooled_subsampled
models=(GRUModel lr lgbm sofa qsofa sirs mews news) #AttentionModel
for model in ${models[@]}; do
    for dataset in ${datasets[@]}; do
        python src/evaluation/pool_predictions.py \
            --dataset_eval $dataset \
            --output_path $output_path \
            --model $model &
    done
done
    
