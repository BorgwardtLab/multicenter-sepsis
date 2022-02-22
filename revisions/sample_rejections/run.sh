#!/bin/bash
# input predictions to consider:
fname=j76ft4wm_EICU_subsample_0.json
input_pred_file=results/evaluation_test/prediction_output_subsampled/${fname}
input_eval_file=results/evaluation_test/evaluation_output_subsampled/${fname}

output_dir=results/sample_rejections
base_dir=${output_dir}/evaluation_test
pred_file=${base_dir}/prediction_output_subsampled/${fname}
eval_file=${base_dir}/evaluation_output_subsampled/${fname}
earliness=median
level=pat
thres=0.8

# uncertainty band:
lower=20
upper=80

plot_path=${base_dir}/plots_${lower}_${upper}

mkdir -p $plot_path

# Apply mask:
    python revisions/sample_rejections/main.py \
        --input $input_pred_file \
        --output_dir $output_dir \
        --lower=$lower \
        --upper=$upper 

# Patient-based Evaluation (on subsample):
    python -m src.evaluation.patient_evaluation \
        --input-file $pred_file \
        --output-file $eval_file \
        --n_jobs=1 \
        --force \
        --cost 5 

# Plotting:
    # masked: 
    python -m scripts.plots.plot_patient_eval \
        --input_path $eval_file  \
        --output_path $plot_path \
        --earliness-stat $earliness \
        --predictions_path $pred_file \
        --level $level \
        --recall_thres $thres

#    # raw input file for comparison:
#    
#    plot_path=${base_dir}/plots_baseline
#    mkdir -p $plot_path
#    python -m scripts.plots.plot_patient_eval \
#        --input_path $input_eval_file  \
#        --output_path $plot_path \
#        --earliness-stat $earliness \
#        --predictions_path $input_pred_file \
#        --level $level \
#        --recall_thres $thres
#
