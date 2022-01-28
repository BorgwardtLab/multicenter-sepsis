#!/bin/bash
run_id=$1 #sepsis/mc-sepsis/69672hhb  #sepsis/mc-sepsis/69672hhb
wandb_path=sepsis/mc-sepsis
wandb_run=${wandb_path}/$run_id
#parallel_tasks=5

base_dir=results #results/pooled #results/feature_ablation #results
split=train #test
pred_path=${base_dir}/evaluation_${split}/prediction_output
mkdir -p $pred_path

datasets=(AUMC) #(AUMC MIMIC Hirid EICU Physionet2019) #MIMIC_LOCF 
for dataset in ${datasets[@]}; do
    pred_file=${pred_path}/${run_id}_${dataset}.json
    sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 4G  -J 'mcsepsis' --wrap "pipenv run python -m src.torch.eval_model_wandb ${wandb_run} --dataset ${dataset} --split ${split} --output ${pred_file}"
done
#non pooled: 4CPU 4G
#MIMIC_LOCF for comparing raw, counts, and locf)
