"""Plot patient-based evaluation.

This script plots the patient-based evaluation information. It requires
access to evaluation files and prediction files.

Example call:

    python -m scripts.plots.plot_patient_eval \
        --input_path evaluations.json         \
        --predictions_path predictions.json   \
        --output_path /tmp/

This will create a plot in `tmp`.
"""

import argparse
import itertools
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np


def plot_curves(df, ax, names={}):
    """Plot curve-based evaluation."""
    # We will need to mirror this axis later on.
    ax1 = ax
    model = names['model']
    train_dataset = names['train_dataset']
    eval_dataset = names['eval_dataset']
    task = names['task']
    ax1.set_title(f'{model}, trained for {task} on {train_dataset}, applied to {eval_dataset}', fontsize=17)
    ax1.set_xlabel('Decision threshold', fontsize=14)
    ax1.set_ylabel('Score', fontsize=14, color='green')
    ax1 = sns.lineplot(x='thres', y='pat_precision', data = df,
            label='precision', color='darkgreen', ax=ax1)
    ax1 = sns.lineplot(x='thres', y='pat_recall', data = df,
            label='recall', color='lightgreen', ax=ax1) #tab:green
    ax1 = sns.lineplot(x='thres', y='physionet2019_utility', data = df,
            label='utility', color='black', ax=ax1)

    ax2 = ax1.twinx()
    ax2.set_ylabel(f'Earliness: {earliness_stat} #hours\nbefore onset',
                   fontsize=8, color='red')
    ax2 = sns.lineplot(x='thres', y=earliness, data = df,
            label='earliness', color='red', ax=ax2)

    plt.xticks(np.linspace(df['thres'].min(), df['thres'].max(), num=10))
    ax1.set_yticks(np.arange(0,1, step=0.1))

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right'), #bbox_to_anchor=(0.05,0.85)) #upper left
    ax1.get_legend().remove()

    return ax.get_xlim()


def plot_scores(df, scores, ax, **kwargs):
    """Plot prediction scores."""
    ax.set_title('Prediction scores', fontsize=17)
    ax = sns.histplot(scores, ax=ax, **kwargs)
    ax.set_xticks(np.linspace(df['thres'].min(), df['thres'].max(), num=10))

    # TODO: hacky, but it works for now and ensures that the axes are
    # aligned. Note that `sharex` does not work because the last plot
    # only contains text.
    ax.set_xlim((xmin, xmax))


def plot_info(df, ax, recall_threshold=0.90):
    """Plot information textbox."""
    # Get precision at the pre-defined recall threshold (typically 90%,
    # but other values might be interesting).
    greater = df['pat_recall'] > recall_threshold
    index = df['pat_recall'][greater].argmin()
    info = df.loc[index]

    earliness = f'earliness_{args.earliness_stat}'

    textbox = '\n'.join((
        f'Patient-based Recall:    {info["pat_recall"]:5.2f}',
        f'Patient-based Precision: {info["pat_precision"]:5.2f}',
        f'Threshold:               {info["thres"]:5.4f}',
        f'Earliness {args.earliness_stat}:           {info[earliness]:5.2f}',
    ))

    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.text(
        0.0, 0.50,
        textbox,
        transform=ax.transAxes,
        fontsize=12,
        family='monospace',
        ha='left',
        va='center'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='results/evaluation/patient_eval_lgbm_aumc_aumc.json')
    parser.add_argument('--predictions_path', required=True)
    parser.add_argument('--output_path', default='results/evaluation/plots')
    parser.add_argument('--earliness-stat', default='mean')
    parser.add_argument('-s', '--show', action='store_true')

    args = parser.parse_args()
    input_path = args.input_path
    with open(input_path, 'r') as f:
        d = json.load(f)
    df = pd.DataFrame(d)

    with open(args.predictions_path, 'r') as f:
        d = json.load(f)
        scores = list(itertools.chain.from_iterable(d['scores']))
        names = {}
        names['model'] = d['model']
        names['train_dataset'] = d['dataset_train']
        names['eval_dataset'] = d['dataset_eval']
        names['task'] = d['task'] 

    earliness_stat = args.earliness_stat
    earliness = f'earliness_{earliness_stat}'

    fig, axes = plt.subplots(nrows=3, figsize=(9, 10), squeeze=True) #6,7

    # for setting title:
    xmin, xmax = plot_curves(df, axes[0], names)
    plot_scores(df, scores, axes[1])
    plot_info(df, axes[2])

    out_file = os.path.split(input_path)[-1].split('.')[0] + '_' + earliness + '.png' 
    plt.tight_layout()
    plt.savefig( os.path.join(args.output_path, out_file))

    if args.show:
        plt.show()
