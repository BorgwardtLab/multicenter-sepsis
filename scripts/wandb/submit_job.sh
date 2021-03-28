#!/bin/bash
sweep_id=$1 #sepsis/mc-sepsis/69672hhb  #sepsis/mc-sepsis/69672hhb
parallel_tasks=5
sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 6G -J 'mcsepsis' --array=3-20%$parallel_tasks --wrap "pipenv run wandb agent --count 1 ${sweep_id} "
# non-EICU: 10 parallel, 4cpu 4G
# EICU: 5 parallel, 4cpu 6G
#export CUDA_VISIBLE_DEVICES=() 
#sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 4G -J 'mcsepsis' --wrap "unset CUDA_VISIBLE_DEVICES; pipenv run wandb agent --count 1 ${sweep_id} "

#export CUDA_VISIBLE_DEVICES=(); 
