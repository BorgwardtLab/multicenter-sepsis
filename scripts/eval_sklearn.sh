#!/bin/bash
input_dir=$1 # path to hyperparameter search results e.g. hypersearch10_regression/
#output_dir=$2 #intermediate results are written folders of this name

base_dir=results
eval_dir=${base_dir}/evaluation

train_datasets=(aumc hirid mimic eicu) #aumc eicu) #aumc #mimic
eval_datasets=(aumc hirid mimic eicu) #aumc #mimic
methods=(lgbm sofa qsofa sirs mews news) #sofa
task=classification 
feature_set=middle #large
cost=5 #lambda cost
n_iter=50 #50 iterations of hypersearch used
earliness=median
thres=0.8

input_features=datasets/{}/data/parquet/features_${feature_set}

for method in ${methods[@]}; do
    for rep in {0..4}; do
        for train_dataset in ${train_datasets[@]}; do
            for eval_dataset in ${eval_datasets[@]}; do
                output_dir=${method}_${train_dataset}_${eval_dataset}_${task}_${feature_set}_cost_${cost}_${n_iter}_iter_rep_${rep} 
                # here we write out the predictions as json
                pred_file=${eval_dir}/prediction_output/${output_dir}.json
                #pred_file=$pred_dir/${method}_${train_dataset}_${eval_dataset}.json
                # here we write out the evaluation metrics as json
                eval_file=${eval_dir}/evaluation_output/${output_dir}.json

                # Apply pretrained model to validation (or test) split:
                python -m src.sklearn.eval_model \
                    --input_path $input_features \
                    --model_path $input_dir \
                    --output_path $pred_file \
                    --method $method \
                    --task $task \
                    --train_dataset $train_dataset \
                    --eval_dataset $eval_dataset \
                    --feature_set $feature_set \
                    --cost $cost \
                    --repetition_model \
                    --rep $rep

                # Patient-based Evaluation:
                python -m src.evaluation.patient_evaluation \
                    --input-file $pred_file \
                    --output-file $eval_file \
                    --n_jobs=100 \
                    --force \
                    --cost $cost

                # Plot patient-based eval metrics:
                python -m scripts.plots.plot_patient_eval \
                    --input_path $eval_file  \
                    --output_path results/evaluation/plots \
                    --earliness-stat $earliness \
                    --predictions_path $pred_file \
                    --recall_thres $thres
            done
        done
    done
done
