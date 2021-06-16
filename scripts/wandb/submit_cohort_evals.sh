#!/bin/bash
run_id=$1 #sepsis/mc-sepsis/69672hhb  #sepsis/mc-sepsis/69672hhb
wandb_path=sepsis/mc-sepsis
wandb_run=${wandb_path}/$run_id
#parallel_tasks=5

base_dir=results
split=test
cohorts=(surg med)

datasets=(AUMC) #MIMIC 
for cohort in ${cohorts[@]}; do
    pred_path=${base_dir}/evaluation_${split}/cohort_${cohort}/prediction_output
    mkdir -p $pred_path
    for dataset in ${datasets[@]}; do
        pred_file=${pred_path}/${run_id}_${dataset}.json
        sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 4G  -J 'mcsepsis' --wrap "pipenv run python -m src.torch.eval_model_wandb ${wandb_run} --dataset ${dataset} --split ${split} --output ${pred_file} --cohort ${cohort}"
    done
done


