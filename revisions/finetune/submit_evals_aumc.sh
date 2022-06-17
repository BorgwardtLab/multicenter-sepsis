#!/bin/bash

# Finetuning of AUMC models on datasets MIMIC, HIRID, EICU 
id_array=("vts4r2yo ryawirr6 2otu55tx znbezbpw 2p42efwv"
    "pfjw8m8q k1d834qr qlsbhfs9 l13lfp6q 1pm76e6j"
    "o56cjqnj 9hf3h204 636feyw5 w6od7sm3 7ja7spva" 
)
datasets=(MIMIC Hirid EICU) 
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


