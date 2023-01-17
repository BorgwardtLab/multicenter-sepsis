#!/bin/bash

# w/o subsampling (if it doesn't work out of the box)

run_id=$1
base_dir=revisions/results
split=test #validation
eval_dir=${base_dir}/evaluation_${split}/cohort_${cohort}

eval_datasets=(AUMC) #(MIMIC) 
eval_datasets2=(aumc) #(mimic) 
cohorts=(surg med)
cost=5 #lambda cost
earliness=median
level=pat
thres=0.8

for cohort in ${cohorts[@]}; do

    eval_dir=${base_dir}/evaluation_${split}/cohort_${cohort}
    #subsampling_pred_dir=${eval_dir}/prediction_output_subsampled
    #subsampling_eval_dir=${eval_dir}/evaluation_output_subsampled
    pred_path=${eval_dir}/prediction_output
    eval_path=${eval_dir}/evaluation_output
    plot_path=${eval_dir}/plots
    paths=($pred_path $eval_path $plot_path)
    for path in ${paths[@]}; do
        mkdir -p $path
    done 

    for index in ${!eval_datasets[*]}; do
        dataset=${eval_datasets[$index]}

        output_name=${run_id}_${dataset}
        pred_file=${pred_path}/${output_name}.json
        eval_file=${eval_path}/${output_name}.json

        # Patient-based Evaluation (on total dataset):
        python -m src.evaluation.patient_evaluation \
            --input-file $pred_file \
            --output-file $eval_file \
            --n_jobs=1 \
            --force \
            --cost $cost &

    done
done
