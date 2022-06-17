#!/bin/bash

# Finetuning of Hirid models on datasets MIMIC, AUMC, EICU 
id_array=("zjeff5su x71s6ror 2ogexznl 7wjaj2hi s9zy0kus"
    "s0gewgca iam8rtkc jtyzfx4r sg8gixdy 5wfxl9rs"
    "dmn4la7o hvm4ojf4 leookt7g hanbx2nt 5h67dgp6"
)

# mimic [‘zjeff5su’, ‘x71s6ror’, ‘2ogexznl’, ‘7wjaj2hi’, ‘s9zy0kus’]
# aumc [‘s0gewgca’, ‘iam8rtkc’, ‘jtyzfx4r’, ‘sg8gixdy’, ‘5wfxl9rs’]
# eicu [‘dmn4la7o’, hvm4ojf4’, ‘leookt7g’, ‘hanbx2nt’, ‘5h67dgp6’]


datasets=(MIMIC AUMC EICU) 
for index in ${!datasets[*]}; do
    run_ids="${id_array[$index]}"
    dataset=${datasets[$index]}

    for run_id in ${run_ids[@]}; do

        wandb_path=sepsis/mc-sepsis
        wandb_run=${wandb_path}/$run_id

        base_dir=results/finetuning #results/pooled #results/feature_ablation #results
        split=test #validation #test
        pred_path=${base_dir}/evaluation_${split}/prediction_output
        mkdir -p $pred_path

        pred_file=${pred_path}/${run_id}_${dataset}.json
        sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 4G  -J 'mcsepsis' --wrap "pipenv run python -m src.torch.eval_model_wandb ${wandb_run} --dataset ${dataset} --split ${split} --output ${pred_file}"

    done

done


