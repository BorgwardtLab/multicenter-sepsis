import argparse
from collections import defaultdict
import os
import functools
import pandas as pd
import numpy as np
import json

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration as PlattScaling
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

datasets = ['eicu', 'aumc', 'mimic', 'hirid']

def get_eval_data(mapping, model="AttentionModel", dataset="hirid", repetition="rep_0"):
    model_mappings = mapping[model]
    iid_mappings = model_mappings[dataset][dataset]

    evaluation_file = iid_mappings[repetition]

    with open(evaluation_file, "r") as f:
        eval_data = json.load(f)
    return eval_data


def get_labels_and_logits(eval_data):
    # This is only valid for the neural network models as other models might
    # require the computation of probabilities though alternative means.
    logits = np.concatenate([np.array(patient_scores) for patient_scores in eval_data["scores"]], -1)
    labels = np.concatenate([np.array(patient_labels) for patient_labels in eval_data["targets"]], -1)
    return labels, logits

def get_patient_labels_and_logits(eval_data):
    logits = np.array( 
            [to_patient_level(patient_scores, is_score=True) for patient_scores in eval_data["scores"]]
    )
    labels = np.array( 
            [to_patient_level(patient_labels, is_score=False) for patient_labels in eval_data["targets"]]
    )
    return labels, logits

def to_patient_level(x, is_score=True): 
    # just as sanity check: check that no nans in here, otherwise this messes patient-level labelling up
    assert not np.any(np.isnan(x))
    if is_score:
        return np.mean(x) #max
    else:
        return np.any(x).astype(int) 

def run_isotonic_regression(logits, labels):
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0., y_max=1.)
    iso_reg.fit(logits, labels)
    return iso_reg


def run_platt_scaling(logits, labels):
    sigmoid_calibration = PlattScaling()
    sigmoid_calibration.fit(logits, labels)
    return sigmoid_calibration


def main(mapping_file_validation, mapping_file_test, models, eval_files_output, mapping_file_output, calibration_method, level):
    os.makedirs(eval_files_output, exist_ok=True)
    output_mapping = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(mapping_file_validation, 'r') as f:
        mapping_validation = json.load(f)
    with open(mapping_file_test, 'r') as f:
        mapping_test = json.load(f)

    for model in models:
        for dataset in datasets:
            if level == 'patient':
                # pool over repetitions
                pooled_labels = []
                pooled_logits = []
            for repetition in ["rep_0", "rep_1", "rep_2", "rep_3", "rep_4"]:
                print(model, dataset, repetition)
                eval_data = get_eval_data(mapping=mapping_validation, model=model, dataset=dataset, repetition=repetition)
                
                if level == 'patient':
                    labels, logits = get_patient_labels_and_logits(eval_data)
                elif level == 'timepoint':
                    labels, logits = get_labels_and_logits(eval_data)
                else:
                    raise ValueError(f' {level} is not among the valid levels.')
                if calibration_method == 'isotonic_regression':
                    calibration = run_isotonic_regression(logits, labels)
                elif calibration_method == 'platt_scaling':
                    calibration = run_platt_scaling(logits, labels)
                else:
                    raise RuntimeError()

                # Rescale predictions
                eval_data = get_eval_data(mapping=mapping_test, model=model, dataset=dataset, repetition=repetition)
                if level == 'timepoint':
                    eval_data["scores"] = [
                        calibration.predict(np.array(patient_scores)).tolist()
                        for patient_scores in eval_data["scores"]
                    ]
                elif level == 'patient':
                    test_labels, test_logits = get_patient_labels_and_logits(eval_data)
                    test_logits = calibration.predict(test_logits).tolist()
                    eval_data["scores"]  = test_logits 
                    eval_data["targets"] = test_labels.tolist()
                
                    pooled_labels.append(test_labels)
                    pooled_logits.append(np.array(test_logits))
                else:
                    raise ValueError(f'{level} not among valid levels.')
                eval_data["calibration_method"] = calibration_method 
                eval_data["calibration_level"] = level
                output_file = os.path.join(eval_files_output, f"{model}_{dataset}_{repetition}.json")
                with open(output_file, "w") as f:
                    json.dump(eval_data, f)

                output_mapping[model][dataset][dataset][repetition] = output_file
           
            if level == 'patient': 
                # aggregated calibration over repetitions:
                print(f'Aggregating calibrations for {model} on {dataset}')
                eval_data = get_eval_data(mapping=mapping_test, model=model, dataset=dataset, repetition=repetition)
                # Sanity check to ensure that all test data is 100% identical and sorted
                mean_pooled_labels = np.mean(pooled_labels, axis=0)
                for pl in pooled_labels:
                    assert np.array_equal(pl, mean_pooled_labels) 
                mean_pooled_logits = np.mean(pooled_logits, axis=0)
                std_pooled_logits = np.std(pooled_logits, axis=0)
                upper = mean_pooled_logits + std_pooled_logits
                lower = mean_pooled_logits - std_pooled_logits

                eval_data["scores_mean"] = mean_pooled_logits.tolist()
                eval_data["scores_upper"] = upper.tolist()
                eval_data["scores_lower"] = lower.tolist()
                eval_data["targets"] = mean_pooled_labels.tolist() 
                
                output_file = os.path.join(eval_files_output, f"{model}_{dataset}_mean.json")
                with open(output_file, "w") as f:
                    json.dump(eval_data, f)

                output_mapping[model][dataset][dataset]["mean"] = output_file
 
 
    with open(mapping_file_output, "w") as f:
        json.dump(output_mapping, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_val", type=str, required=True)
    parser.add_argument("--mapping_test", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", default=["AttentionModel"])
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_mapping", type=str, required=True)
    parser.add_argument("--calibration_method", type=str, choices=["isotonic_regression", "platt_scaling"])
    parser.add_argument("--level", type=str, choices=["patient", "timepoint"], default="timepoint")

    args = parser.parse_args()

    main(args.mapping_val, args.mapping_test, args.models, args.output_folder, args.output_mapping, args.calibration_method, args.level)
