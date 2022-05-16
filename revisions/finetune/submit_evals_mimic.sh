#!/bin/bash
##run_id=$1 #sepsis/mc-sepsis/69672hhb  #sepsis/mc-sepsis/69672hhb
#run_ids=(w2tm5r46 0am807pj hjgnm8rh vd0yg6lb bvm4zsko) #runs for models trained on MIMIC, finetuned on AUMC (val split)
#datasets=(AUMC) #(AUMC MIMIC Hirid EICU Physionet2019) #MIMIC_LOCF 
#
#for run_id in ${run_ids[@]}; do
#
#    wandb_path=sepsis/mc-sepsis
#    wandb_run=${wandb_path}/$run_id
#    #parallel_tasks=5
#
#    base_dir=results/finetuning #results/pooled #results/feature_ablation #results
#    split=test #validation #test
#    pred_path=${base_dir}/evaluation_${split}/prediction_output
#    mkdir -p $pred_path
#
#    for dataset in ${datasets[@]}; do
#        pred_file=${pred_path}/${run_id}_${dataset}.json
#        sbatch -p gpu --gres=gpu:1 --cpus-per-task 4 --mem-per-cpu 4G  -J 'mcsepsis' --wrap "pipenv run python -m src.torch.eval_model_wandb ${wandb_run} --dataset ${dataset} --split ${split} --output ${pred_file}"
#    done
#    #non pooled: 4CPU 4G
#    #MIMIC_LOCF for comparing raw, counts, and locf)
#
#done

# Remaining models of MIMIC finetuning:
id_array=("2rg0gwsz wbovv3bq 2ov1oq2b 2y6tz6xm ypy4sllk"
    "frgx6qfm swiiispj dilkoi12 vdgbmb8v 8yuw4l4o" 
)
datasets=(Hirid EICU) 
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


