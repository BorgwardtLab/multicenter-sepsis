#!/bin/bash
input_dir=$1 # path to hyperparameter search results e.g. hypersearch10_regression/
#output_dir=$2 #intermediate results are written folders of this name

base_dir=results
eval_dir=${base_dir}/evaluation


train_datasets=(hirid aumc mimic eicu) #aumc #mimic
eval_datasets=(hirid aumc mimic eicu) #aumc #mimic
method=lgbm #lgbm
task=regression #classification 
feature_set=middle #large
cost=5 #lambda cost
n_iter=50 #50 iterations of hypersearch used
earliness=median

for train_dataset in ${train_datasets[@]}; do
    for eval_dataset in ${eval_datasets[@]}; do
        output_dir=${method}_${train_dataset}_${eval_dataset}_${task}_${feature_set}_cost_${cost}_${n_iter}_iter 
        # here we write out the predictions as json
        pred_file=${eval_dir}/prediction_output/${output_dir}.json
        #pred_file=$pred_dir/${method}_${train_dataset}_${eval_dataset}.json
        # here we write out the evaluation metrics as json
        eval_file=${eval_dir}/evaluation_output/${output_dir}.json

        # Apply pretrained model to validation (or test) split:
        python -m src.sklearn.eval_model \
            --model_path $input_dir \
            --output_path $pred_file \
            --method $method \
            --task $task \
            --train_dataset $train_dataset \
            --eval_dataset $eval_dataset \
            --feature_set $feature_set \
            --cost $cost

        # Patient-based Evaluation:
        python -m src.evaluation.patient_evaluation \
            --input-file $pred_file \
            --output-file $eval_file \
            --n_jobs=50 \
            --force \
            --cost $cost

        # Plot patient-based eval metrics:
        python scripts/plots/plot_patient_eval.py \
            --input_path $eval_file  \
            --output_path results/evaluation/plots \
            --earliness-stat $earliness \
            --predictions_path $pred_file
    done
done
