#!/bin/bash
sweep_id=$1 #sepsis/mc-sepsis/69672hhb  #sepsis/mc-sepsis/69672hhb
sweep_path=sepsis/mc-sepsis/${sweep_id}
parallel_tasks=6
sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 6G --exclude=bs-gpu07 -J 'mcsepsis' --array=1-5%$parallel_tasks --wrap "pipenv run wandb agent --count 1 ${sweep_path} "
#sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 6G --nodelist=bs-gpu07 -J 'mcsepsis' --array=1-2%$parallel_tasks --wrap "pipenv run wandb agent --count 1 ${sweep_path} "

# MEMORY:
# non-EICU: 10 parallel, 4cpu 4G
# EICU: 5 parallel, 4cpu 6G

