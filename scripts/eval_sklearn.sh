#!/bin/bash
input_dir=$1 # path to hyperparameter search results e.g. hypersearch10_regression/
output_dir=$2 #intermediate results are written folders of this name

base_dir=results
eval_dir=${base_dir}/evaluation

train_dataset=aumc #mimic
eval_dataset=aumc #mimic
method=lgbm
task=regression
feature_set=middle #large

# here we write out the predictions as json
pred_dir=${eval_dir}/prediction_output/${output_dir}
pred_file=$pred_dir/${method}_${train_dataset}_${eval_dataset}.json
# here we write out the evaluation metrics as json
eval_file=${eval_dir}/evaluation_output/${output_dir}.json

# Apply pretrained model to validation (or test) split:
python -m src.sklearn.eval_model \
    --model_path $input_dir \
    --output_path $pred_dir \
    --method $method \
    --task $task \
    --train_dataset $train_dataset \
    --eval_dataset $eval_dataset \
    --feature_set $feature_set 

# Patient-based Evaluation:
python -m src.evaluation.patient_evaluation \
    --input-file $pred_file \
    --output-file $eval_file \
    --n_jobs=100 \
    --force \
    --task $task

# Plot patient-based eval metrics:
python scripts/plots/plot_patient_eval.py \
    --input_path $eval_file  \
    --output_path results/evaluation/plots
