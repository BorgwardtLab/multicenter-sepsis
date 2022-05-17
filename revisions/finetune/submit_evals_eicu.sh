#!/bin/bash

# Testing EICU models finetuned on datasets MIMIC, AUMC, EICU 
id_array=("licfx74s xa0zjlg1 95dvg2fn mm86lcxh vjnfxnfg"
    "6di88zkf k68r5cn0 nvlc09i5 3v9pbmva qn2tethe"
    "pl1iqije 9wvljri2 l4vl4lyq 47xdrrra k8mnrfsq"
)
 id_array=(
    "k68r5cn0"
)
 
datasets=(MIMIC Hirid AUMC)
datasets=(Hirid)
 
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


