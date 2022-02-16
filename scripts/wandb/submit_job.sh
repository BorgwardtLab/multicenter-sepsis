#!/bin/bash
sweep_id=$1 #sepsis/mc-sepsis/69672hhb  #sepsis/mc-sepsis/69672hhb
sweep_path=sepsis/mc-sepsis/${sweep_id}
parallel_tasks=6 #4
n_runs=58 #3
sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 8G -J 'mcsepsis' --array=1-${n_runs}%$parallel_tasks --wrap "pipenv run wandb agent --count 1 ${sweep_path} "
#sbatch -p gpu --gres=gpu:1 --nice --cpus-per-task 4 --mem-per-cpu 6G --nodelist=bs-gpu07 -J 'mcsepsis' --array=1-2%$parallel_tasks --wrap "pipenv run wandb agent --count 1 ${sweep_path} "
## --exclude=bs-gpu09
## --nice prio down

# MEMORY:
# non-EICU: 10 parallel, 4cpu 4G
# EICU: 5 parallel, 4cpu 6G
# pooled: 3 par, 3 or 4cpu, 8G
# pooled non-EICU: 4cpu 6G
