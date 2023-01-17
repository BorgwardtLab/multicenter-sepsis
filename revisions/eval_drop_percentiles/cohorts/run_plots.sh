
base_path=revisions/results/pooled/evaluation_test

#### Collecting results 
#python -m scripts.plots.gather_data \
#    --input_path ${base_path}/evaluation_output_subsampled \
#    --output_path ${base_path}/plots
#
## ROC plots
python scripts/plots/plot_roc.py \
    --input_path ${base_path}/plots/result_data_subsampled.csv 

# PR  plots
python scripts/plots/plot_pr.py \
    --input_path ${base_path}/plots/result_data_subsampled.csv 

# Scatterplot
python -m scripts.plots.plot_scatterplots \
    ${base_path}/plots/result_data_subsampled.csv \
    --r 0.80 \
    --point-alpha 0.35 \
    --line-alpha 1.0 \
    --output ${base_path}/plots 
