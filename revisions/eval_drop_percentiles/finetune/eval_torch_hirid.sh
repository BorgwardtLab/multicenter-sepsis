#!/bin/bash

# updated prevalence

# Evaluating the predictions of Hirid models on datasets MIMIC, AUMC, EICU 

id_array=("zjeff5su x71s6ror 2ogexznl 7wjaj2hi s9zy0kus"
    "s0gewgca iam8rtkc jtyzfx4r sg8gixdy 5wfxl9rs"
    "dmn4la7o hvm4ojf4 leookt7g hanbx2nt 5h67dgp6"
)

base_dir=results/finetuning #results/pooled #feature_ablation #results 
split=test #validation
eval_dir=${base_dir}/evaluation_${split}

eval_datasets=(MIMIC AUMC EICU) #(AUMC MIMIC Hirid EICU Physionet2019) #MIMIC_LOCF
eval_datasets2=(mimic aumc eicu) #(aumc mimic hirid eicu physionet2019) #sklearn formatting  (physionet2019) #

cost=5 #lambda cost
earliness=median
level=pat
thres=0.8
subsampling_pred_dir=${eval_dir}/prediction_output_subsampled
subsampling_eval_dir=${eval_dir}/evaluation_output_subsampled
pred_path=${eval_dir}/prediction_output
eval_path=${eval_dir}/evaluation_output
plot_path=${eval_dir}/plots
paths=($subsampling_pred_dir $subsampling_eval_dir $pred_path $eval_path $plot_path)
for path in ${paths[@]}; do
    mkdir -p $path
done 


# for run_id in ${run_ids[@]}; do

for index in ${!eval_datasets[*]}; do
    run_ids=${id_array[$index]}
    dataset=${eval_datasets[$index]}
    sklearn_dataset=${eval_datasets2[$index]}
    echo Evaluating model finetuned on ${dataset} with runs ${run_ids}
    for run_id in $run_ids; do

        output_name=${run_id}_${dataset}
        pred_file=${pred_path}/${output_name}.json
        eval_file=${eval_path}/${output_name}.json

        # Subsampling 10 times at harmonized prevalence 
        python src/evaluation/subsampling.py \
            --input-file $pred_file \
            --output-dir $subsampling_pred_dir \
            --subsampling-file config/splits/subsamples_${sklearn_dataset}_prev_0.188.json

        for subsample in {0..9}; do
            subsampled_predictions=${subsampling_pred_dir}/${output_name}_subsample_${subsample}.json
            subsampled_evaluations=${subsampling_eval_dir}/${output_name}_subsample_${subsample}.json 

            # Patient-based Evaluation (on subsample):
            python -m src.evaluation.patient_evaluation \
            --input-file $subsampled_predictions \
            --output-file $subsampled_evaluations \
            --n_jobs=1 \
            --force \
            --cost $cost \
            --drop_percentiles &
        done

        # Patient-based Evaluation (on total dataset):
        python -m src.evaluation.patient_evaluation \
            --input-file $pred_file \
            --output-file $eval_file \
            --n_jobs=1 \
            --force \
            --cost $cost \
            --drop_percentiles

        # Plot patient-based eval metrics:
        python -m scripts.plots.plot_patient_eval \
            --input_path $eval_file  \
            --output_path $plot_path \
            --earliness-stat $earliness \
            --predictions_path $pred_file \
            --level $level \
            --recall_thres $thres
    done
done
