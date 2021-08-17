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


def optim_step(optimizer, logits, labels, temperature):
    optimizer.zero_grad()
    loss = F.binary_cross_entropy_with_logits(logits/temperature.expand(logits.shape), labels)
    loss.backward()
    return loss

def run_isotonic_regression(logits, labels):
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0., y_max=1.)
    iso_reg.fit(logits, labels)
    return iso_reg


def run_platt_scaling(logits, labels):
    sigmoid_calibration = PlattScaling()
    sigmoid_calibration.fit(logits, labels)
    return sigmoid_calibration


def main(mapping_file_validation, mapping_file_test, models, eval_files_output, mapping_file_output, calibration_method):
    os.makedirs(eval_files_output, exist_ok=True)
    output_mapping = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(mapping_file_validation, 'r') as f:
        mapping_validation = json.load(f)
    with open(mapping_file_test, 'r') as f:
        mapping_test = json.load(f)

    for model in models:
        for dataset in datasets:
            for repetition in ["rep_0", "rep_1", "rep_2", "rep_3", "rep_4"]:
                print(model, dataset, repetition)
                eval_data = get_eval_data(mapping=mapping_validation, model=model, dataset=dataset, repetition=repetition)
                labels, logits = get_labels_and_logits(eval_data)
                if calibration_method == 'isotonic_regression':
                    calibration = run_isotonic_regression(logits, labels)
                elif calibration_method == 'platt_scaling':
                    calibration = run_platt_scaling(logits, labels)
                else:
                    raise RuntimeError()

                # Rescale predictions
                eval_data = get_eval_data(mapping=mapping_test, model=model, dataset=dataset, repetition=repetition)
                eval_data["scores"] = [
                    calibration.predict(np.array(patient_scores)).tolist()
                    for patient_scores in eval_data["scores"]
                ]
                eval_data["calibration_method"] = calibration_method
                output_file = os.path.join(eval_files_output, f"{model}_{dataset}_{repetition}.json")
                with open(output_file, "w") as f:
                    json.dump(eval_data, f)

                output_mapping[model][dataset][dataset][repetition] = output_file

    with open(mapping_file_output, "w") as f:
        json.dump(output_mapping, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_val", type=str, required=True)
    parser.add_argument("--mapping_test", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", default=["AttentionModel", "GRUModel"])
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_mapping", type=str, required=True)
    parser.add_argument("--calibration_method", type=str, choices=["isotonic_regression", "platt_scaling"])
    args = parser.parse_args()

    main(args.mapping_val, args.mapping_test, args.models, args.output_folder, args.output_mapping, args.calibration_method)
