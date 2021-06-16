#!/bin/bash

base_dir=results
split=test
eval_dir=${base_dir}/evaluation_${split}/prediction_pooled_subsampled

eval_datasets=(eicu aumc hirid mimic) #(aumc hirid mimic eicu)
method=$1 #(lgbm sofa qsofa sirs mews news)) #lgbm sofa
task=classification 
feature_set=middle #large
cost=5 #lambda cost
earliness=median
thres=0.8


aggregations=(max) #mean
 
for agg in ${aggregations[@]}; do
    pred_path=${eval_dir}/${agg}/prediction_output
    eval_path=${eval_dir}/${agg}/evaluation_output
    paths=($pred_path $eval_path)
    for path in ${paths[@]}; do
        mkdir -p $path 
    done

    for rep in {0..4}; do #{0..4}
        for eval_dataset in ${eval_datasets[@]}; do
            for subsample in {0..9}; do #9
                output_name=preds_${agg}_pooled_${method}_${eval_dataset}_rep_${rep}_subsample_${subsample}.json 
                pred_file=${pred_path}/${output_name}
                # here we write out the evaluation metrics as json
                eval_file=${eval_path}/${output_name}

                # Patient-based Evaluation (on subsample):
                python -m src.evaluation.patient_evaluation \
                --input-file $pred_file \
                --output-file $eval_file \
                --n_jobs=1 \
                --force \
                --cost $cost &
            done
            wait
        done
    done
done
