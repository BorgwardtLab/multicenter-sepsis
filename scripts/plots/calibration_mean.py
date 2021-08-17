import argparse
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import sklearn.calibration as calibration

# MAPPING_FILE = "/links/groups/borgwardt/Projects/sepsis/multicenter-sepsis/results/evaluation_test/prediction_mapping.json"
datasets = ['eicu', 'aumc', 'physionet2019', 'mimic', 'hirid']
datasets = ['eicu', 'aumc', 'mimic', 'hirid']


def get_eval_data(mapping_file, model, dataset, repetition="mean"):
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    model_mappings = mapping[model]
    iid_mappings = model_mappings[dataset][dataset]

    evaluation_file = iid_mappings[repetition]

    with open(evaluation_file, "r") as f:
        eval_data = json.load(f)
    return eval_data

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ensure_zero_one(probs):
    # fix numerical issue due to std leading to invalid probs.
    assert probs.min() > -0.1 
    assert probs.max() < 1.1
    return np.clip(probs, 0,1)

def get_labels_and_probabilites(eval_data, use_sigmoid=True, level='patient', suffix='_mean'):
    # This is only valid for the neural network models as other models might
    # require the computation of probabilities though alternative means.
    assert level == 'patient'
    labels = np.array(eval_data["targets"]) # we assume that in the first calibration script, the targets are overwritten
    # with patient labels 
    probabilities = np.array(eval_data["scores" + suffix])

    if use_sigmoid:
        probabilities = sigmoid(probabilities)
    return labels, probabilities


def main(mapping_file, model, output, use_sigmoid=True, level='timepoint', nbins=20): #nbins=100
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ## ax2 = plt.subplot2grid((3, 1), (2, 0))

    for dataset in datasets:
        prob_preds = []
        prob_trues = []
        prob_histograms = []
        repetition = "mean"
        eval_data = get_eval_data(mapping_file, model=model, dataset=dataset, repetition=repetition)
        labels, probabilities = get_labels_and_probabilites(eval_data, use_sigmoid=use_sigmoid, level=level, suffix='_mean')
        prob_true_mean, prob_pred_mean = calibration.calibration_curve(labels, probabilities, n_bins=nbins)
        x = np.linspace(0., 1., nbins)
    
        _, probabilities_lower = get_labels_and_probabilites(eval_data, use_sigmoid=use_sigmoid, level=level, suffix='_lower')
        _, probabilities_upper = get_labels_and_probabilites(eval_data, use_sigmoid=use_sigmoid, level=level, suffix='_upper')
        probabilities_lower = ensure_zero_one(probabilities_lower)
        probabilities_upper = ensure_zero_one(probabilities_upper)

        prob_true_lower, prob_pred_lower = calibration.calibration_curve(labels, probabilities_lower, n_bins=nbins)
        prob_true_upper, prob_pred_upper = calibration.calibration_curve(labels, probabilities_upper, n_bins=nbins)

        ## prob_trues.append(np.interp(x, prob_pred, prob_true))
        ## prob_preds.append(x)
        ## hist, edges = np.histogram(probabilities, bins=nbins, range=(0., 1.))
        ## prob_histograms.append(hist)

        ## prob_preds = np.stack(prob_preds, -1)
        ## prob_preds_mean = np.mean(prob_preds, -1) # These should all be the same
        ## prob_trues = np.stack(prob_trues, -1)
        ## prob_histograms = np.stack(prob_histograms, -1)
        ## prob_trues_mean = np.mean(prob_trues, -1)
        ## prob_trues_std = np.std(prob_trues, -1)
        ## prob_histograms_mean = np.mean(prob_histograms, -1)
        ## prob_histograms_std = np.std(prob_histograms, -1)

        p = ax1.plot(prob_pred_mean, prob_true_mean, "-", label=dataset)[0]
        color = p.get_color()
        # ax1.fill_between(prob_pred_mean, prob_true_lower, prob_true_upper, color=color, alpha=0.3)

        ## # In order to allow shaded areas we need to build our own histogram.
        ## histogram_x = np.repeat(edges, 2)[1:-1]
        ## histogram_y_mean = np.repeat(prob_histograms_mean, 2)
        ## histogram_y_std = np.repeat(prob_histograms_std, 2)
        ## ax2.plot(histogram_x, histogram_y_mean, color=color)
        ## ax2.fill_between(histogram_x, histogram_y_mean-histogram_y_std, histogram_y_mean+histogram_y_std, color=color, alpha=0.3)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([0., 1.])
    ax1.legend(loc="upper left")
    ax1.set_title('Calibration plots  (reliability curve)')

    ## ax2.set_xlabel("Mean predicted value")
    ## ax2.set_ylabel("Count")
    ## ax2.set_xlim([0., 1.])

    plt.tight_layout()
    plt.savefig(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--no_sigmoid', action='store_true', default=False)
    parser.add_argument("--level", type=str, choices=["patient"], default="patient") # currently only patient-level

    args = parser.parse_args()
    main(args.mapping_file, args.model, args.output, use_sigmoid=not args.no_sigmoid, level=args.level)
