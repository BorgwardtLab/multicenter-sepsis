# Get prediction_mapping 
python -m scripts.map_model_to_result_files \
    results/finetuning/evaluation_test/prediction_output/ \
    --output_path results/finetuning/evaluation_test/prediction_mapping.json \
    --overwrite


# Actual pooling:
