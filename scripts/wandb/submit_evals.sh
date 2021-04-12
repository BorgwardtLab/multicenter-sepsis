#!/bin/bash
run_id=$1 #sepsis/mc-sepsis/69672hhb  #sepsis/mc-sepsis/69672hhb
wandb_path=sepsis/mc-sepsis
wandb_run=${wandb_path}/$run_id
#parallel_tasks=5
split=validation
datasets=(AUMC MIMIC EICU Hirid)
for dataset in ${datasets[@]}; do
    pred_file=results/evaluation/prediction_output/${run_id}_${dataset}.json
    sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 6G -J 'mcsepsis' --wrap "pipenv run python -m src.torch.eval_model_wandb ${wandb_run} --dataset ${dataset} --split ${split} --output ${pred_file}"
done
# MEMORY:
# non-EICU: 10 parallel, 4cpu 4G
# EICU: 5 parallel, 4cpu 6G

