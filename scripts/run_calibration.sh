#!/bin/bash
# Run calibration
mkdir -p results/calibration
pipenv run python scripts/calibrate_model_platt_isotonic.py --mapping_val results/evaluation_validation/prediction_mapping.json  --mapping_test results/evaluation_test/prediction_mapping.json --output_folder results/calibration/platt_scaling --output_mapping results/calibration/platt_mapping.json --calibration_method platt_scaling
pipenv run python scripts/calibrate_model_platt_isotonic.py --mapping_val results/evaluation_validation/prediction_mapping.json  --mapping_test results/evaluation_test/prediction_mapping.json --output_folder results/calibration/isotonic_regression --output_mapping results/calibration/isotonic_mapping.json --calibration_method isotonic_regression
pipenv run python scripts/calibrate_model_temperature_scaling.py --mapping_val results/evaluation_validation/prediction_mapping.json  --mapping_test results/evaluation_test/prediction_mapping.json --output_folder results/calibration/temperature_scaling --output_mapping results/calibration/temperature_mapping.json

# Run plots
mkdir -p results/calibration/plots
pipenv run python scripts/plots/calibration.py --mapping_file results/evaluation_test/prediction_mapping.json --model AttentionModel --output results/calibration/plots/AttentionModel.pdf
pipenv run python scripts/plots/calibration.py --mapping_file results/evaluation_test/prediction_mapping.json --model GRUModel --output results/calibration/plots/GRUModel.pdf
pipenv run python scripts/plots/calibration.py --mapping_file results/calibration/platt_mapping.json --model AttentionModel --output results/calibration/plots/AttentionModel_platt.pdf --no_sigmoid
pipenv run python scripts/plots/calibration.py --mapping_file results/calibration/platt_mapping.json --model GRUModel --output results/calibration/plots/GRUModel_platt.pdf --no_sigmoid
pipenv run python scripts/plots/calibration.py --mapping_file results/calibration/isotonic_mapping.json --model AttentionModel --output results/calibration/plots/AttentionModel_isotonic.pdf --no_sigmoid
pipenv run python scripts/plots/calibration.py --mapping_file results/calibration/isotonic_mapping.json --model GRUModel --output results/calibration/plots/GRUModel_isotonic.pdf --no_sigmoid
pipenv run python scripts/plots/calibration.py --mapping_file results/calibration/temperature_mapping.json --model AttentionModel --output results/calibration/plots/AttentionModel_temperature.pdf --no_sigmoid
pipenv run python scripts/plots/calibration.py --mapping_file results/calibration/temperature_mapping.json --model GRUModel --output results/calibration/plots/GRUModel_temperature.pdf --no_sigmoid