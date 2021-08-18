import argparse
from collections import defaultdict
import os
import functools
import pandas as pd
import numpy as np
import json

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
        return np.mean(x)
    else:
        return np.any(x).astype(float) 


def optim_step(optimizer, logits, labels, temperature):
    optimizer.zero_grad()
    loss = F.binary_cross_entropy_with_logits(logits/temperature.expand(logits.shape), labels)
    loss.backward()
    return loss

def run_temperature_scaling(logits, labels):
    logits, labels = torch.tensor(logits), torch.tensor(labels)
    temp = torch.nn.Parameter(torch.full((1,), 1.5))
    optimizer = torch.optim.LBFGS([temp], lr=0.01, max_iter=300) #lr=0.5 
    step = functools.partial(optim_step, optimizer, logits, labels, temp)
    print('BCE loss prior to temperature scaling:', F.binary_cross_entropy_with_logits(logits, labels).item())
    optimizer.step(step)
    print('BCE loss after temperature scaling:', F.binary_cross_entropy_with_logits(logits/temp.expand(logits.shape), labels).item())
    return temp.item()


def main(mapping_file_validation, mapping_file_test, models, eval_files_output, mapping_file_output, level):
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
                if level == 'patient':
                    labels, logits = get_patient_labels_and_logits(eval_data)
                elif level == 'timepoint':
                    labels, logits = get_labels_and_logits(eval_data)
                else:
                    raise ValueError(f' {level} is not among the valid levels.')

                temperature = run_temperature_scaling(logits, labels)
                print("Temperature:", temperature)

                # Rescale predictions
                eval_data = get_eval_data(mapping=mapping_test, model=model, dataset=dataset, repetition=repetition)
                if level == 'timepoint':
                    eval_data["scores"] = [
                    (np.array(patient_scores)/temperature).tolist()
                    for patient_scores in eval_data["scores"]
                    ]
                elif level == 'patient':
                    test_labels, test_logits = get_patient_labels_and_logits(eval_data)
                    eval_data["scores"] = (test_logits/temperature).tolist()
                    eval_data["targets"] = test_labels.tolist()
                else:
                    raise ValueError(f'{level} not among valid levels.')
                eval_data["calibration_method"] = "temperature_scaling"
                eval_data["calibration_level"] = level

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
    parser.add_argument("--models", type=str, nargs="+", default=["AttentionModel"])
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_mapping", type=str, required=True)
    parser.add_argument("--level", type=str, choices=["patient", "timepoint"], default="timepoint")

    args = parser.parse_args()

    main(args.mapping_val, args.mapping_test, args.models, args.output_folder, args.output_mapping, args.level)
