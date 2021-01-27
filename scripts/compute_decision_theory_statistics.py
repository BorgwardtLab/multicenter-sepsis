#!/usr/bin/env python
"""Compute the statistics need for Bayesian decision theory."""
import argparse
import pickle
import os
import pandas as pd
import numpy as np
from collections import Counter


def load_pickle(filename):
    """ Basic pickle loading function """
    with open(filename, "rb") as file:
        obj = pickle.load(file)
    return obj


def get_deltat_distribution(patient):
    if hasattr(patient["time"], "iloc"):
        onset = patient["time"].iloc[np.argmax(patient["sep3"])]
    elif hasattr(patient["time"], "__len__"):
        onset = patient["time"][np.argmax(patient["sep3"])]
    else:
        return np.array([0])
    return np.array(patient["time"] - onset)


def compute_statistics(df):
    delta_t_dist = Counter()
    has_sepsis = X.groupby(X.index).apply(lambda df: (df["sep3"] == 1).any())
    sepsis_patients = has_sepsis.index[has_sepsis]
    for patient in sepsis_patients:
        delta_t = get_deltat_distribution(X.loc[patient])
        delta_t_dist.update(delta_t)

    return delta_t_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="path from dataset to pickled df file to inspect",
        default="data/sklearn/processed",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use, when setting `all`, all datasets are iterated over",
        default="physionet2019",
    )
    args = parser.parse_args()
    dataset = args.dataset

    # check if looping over all splits and datasets or just single provided one

    if dataset == "all":
        datasets = ["physionet2019", "mimic3", "eicu", "hirid", "aumc"]
    else:
        datasets = [dataset]

    results = {}
    for dataset in datasets:
        path = os.path.join("datasets", dataset, args.path)
        filtered_path = os.path.join(path, "X_filtered_train.pkl")
        X = load_pickle(filtered_path)
        X = X[["time", "sep3"]]  # We only need time and the label
        delta_t_dist = compute_statistics(X)
        keys = list(delta_t_dist.keys())
        keys.sort()
        values = [delta_t_dist[key] for key in keys]
        import matplotlib.pyplot as plt

        plt.plot(keys, values)
        plt.title(dataset)
        plt.xlabel("$\Delta t$ from onset")
        plt.ylabel("Frequency")
        plt.savefig(f"{dataset}.png")
        plt.close()
