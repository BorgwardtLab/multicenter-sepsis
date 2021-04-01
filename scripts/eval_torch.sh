#!/bin/bash

run_id=$1
base_dir=results
eval_dir=${base_dir}/evaluation
eval_datasets=(AUMC MIMIC EICU Hirid)

cost=5 #lambda cost
earliness=median


for dataset in ${eval_datasets[@]}; do
    pred_file=${eval_dir}/prediction_output/${run_id}_${dataset}.json
    eval_file=${eval_dir}/evaluation_output/${run_id}_${dataset}.json
 
    # Patient-based Evaluation:
    python -m src.evaluation.patient_evaluation \
        --input-file $pred_file \
        --output-file $eval_file \
        --n_jobs=100 \
        --force \
        --cost $cost

    # Plot patient-based eval metrics:
    python scripts/plots/plot_patient_eval.py \
        --input_path $eval_file  \
        --output_path results/evaluation/plots \
        --earliness-stat $earliness \
        --predictions_path $pred_file
done
