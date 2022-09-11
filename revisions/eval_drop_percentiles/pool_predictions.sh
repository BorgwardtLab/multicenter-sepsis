#!/bin/bash

# get pred mapping:
base_path=revisions/results/evaluation_test
pred_ss_path=${base_path}/prediction_output_subsampled/
mapping_file=${base_path}/prediction_subsampled_mapping.json
  
#python -m scripts.map_model_to_result_files $pred_ss_path \
#    --output_path $mapping_file \
#    --overwrite 

datasets=(hirid eicu aumc mimic)
output_path=${base_path}/prediction_pooled_subsampled
models=(AttentionModel GRUModel lr lgbm sofa qsofa sirs mews news)
for model in ${models[@]}; do
    for dataset in ${datasets[@]}; do
        python src/evaluation/pool_predictions.py \
            --mapping_file $mapping_file \
            --dataset_eval $dataset \
            --output_path $output_path \
            --model $model &
    done
done
    
