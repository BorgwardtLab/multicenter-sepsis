#!/bin/bash

split=test
base_dir=results/finetuning/evaluation_${split}
eval_dir=${base_dir}/evaluation_output_subsampled
plot_path=${base_dir}/plots

#python -m scripts.plots.gather_data --input_path $eval_dir \
#    --output_path $plot_path 
#
python revisions/finetune/plot_roc.py --input_path ${plot_path}/result_data_subsampled.csv 

python -m revisions.finetune.plot_scatterplots ${plot_path}/result_data_subsampled.csv \
     --r 0.80 --point-alpha 0.35 --line-alpha 1.0 --output $plot_path 
