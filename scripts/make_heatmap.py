"""Create heatmap from CSV of results."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import itertools
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument(
        '-m', '--metric',
        type=str,
        default='auc_mean',
        help='Metric to use for edge weights'
    )

    args = parser.parse_args()

    df = pd.read_csv(args.INPUT)

    df = df.pivot(
        'train_dataset',
        columns='eval_dataset',
        values=args.metric
    )

    for dataset in df.columns:
        df[dataset][dataset] = np.nan

    sns.heatmap(df, cmap='RdYlGn', annot=True)

    plt.tick_params(
        axis='both',
        which='major',
        labelleft=True,
        labelright=True,
        labelbottom=True,
        labeltop=True
    )
    plt.show()
