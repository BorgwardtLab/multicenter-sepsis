#!/bin/bash
# Run calibration
mkdir -p results/calibration

## raw scores (on pat level)
#pipenv run python scripts/calibrate_model_platt_isotonic.py --mapping_val results/evaluation_validation/prediction_mapping.json  \
#                                                            --mapping_test results/evaluation_test/prediction_mapping.json \
#                                                            --output_folder results/calibration/raw \
#                                                            --output_mapping results/calibration/raw_mapping_patient_level.json \
#                                                            --calibration_method raw \
#                                                            --level patient
## isotonic calib.
#pipenv run python scripts/calibrate_model_platt_isotonic.py --mapping_val results/evaluation_validation/prediction_mapping.json  \
#                                                            --mapping_test results/evaluation_test/prediction_mapping.json \
#                                                            --output_folder results/calibration/isotonic_regression \
#                                                            --output_mapping results/calibration/isotonic_mapping_patient_level.json \
#                                                            --calibration_method isotonic_regression \
#
#                                                            --level patient

# platt calib.
#pipenv run python scripts/calibrate_model_platt_isotonic.py --mapping_val results/evaluation_validation/prediction_mapping.json  \
#                                                            --mapping_test results/evaluation_test/prediction_mapping.json \
#                                                            --output_folder results/calibration/platt_scaling \
#                                                            --output_mapping results/calibration/platt_mapping_patient_level.json \
#                                                            --calibration_method platt_scaling \
#                                                            --level patient

## temperature scaling:
#pipenv run python scripts/calibrate_model_temperature_scaling.py    --mapping_val results/evaluation_validation/prediction_mapping.json  \
#                                                                    --mapping_test results/evaluation_test/prediction_mapping.json \
#                                                                    --output_folder results/calibration/temperature_scaling \
#                                                                    --output_mapping results/calibration/temperature_mapping_patient_level.json \
#                                                                    --level patient
#


# Run plots
mkdir -p results/calibration/plots

## raw plot
pipenv run python scripts/plots/calibration.py --mapping_file results/calibration/raw_mapping_patient_level.json --model AttentionModel --output results/calibration/plots/AttentionModel_raw.pdf --level patient
#
## isotonic plot
pipenv run python scripts/plots/calibration.py  --mapping_file results/calibration/isotonic_mapping_patient_level.json \
                                                --model AttentionModel \
                                                --output results/calibration/plots/AttentionModel_isotonic.pdf \
                                                --level patient \
                                                --no_sigmoid

# platt plot
pipenv run python scripts/plots/calibration.py  --mapping_file results/calibration/platt_mapping_patient_level.json \
                                                --model AttentionModel \
                                                --output results/calibration/plots/AttentionModel_platt.pdf \
                                                --level patient \
                                                --no_sigmoid
# temp plot
# pipenv run python scripts/plots/calibration.py  --mapping_file results/calibration/temperature_mapping_patient_level.json \
#                                                --model AttentionModel \
#                                                --output results/calibration/plots/AttentionModel_temperature.pdf \
#                                                --level patient 
#
