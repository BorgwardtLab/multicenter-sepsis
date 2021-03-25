"""Plot patient-based evaluation."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np 


def plot_curves(df, ax):
    """Plot curve-based evaluation."""
    # We will need to mirror this axis later on.
    ax1 = ax
    ax1.set_title('Patient-based Evaluation', fontsize=19)
    ax1.set_xlabel('Decision threshold', fontsize=16)
    ax1.set_ylabel('Score', fontsize=16, color='green')
    #ax1 = sns.lineplot(x='thres', data = df[['thres', 'pat_recall','pat_precision']])
    ax1 = sns.lineplot(x='thres', y='pat_precision', data = df,
            label='precision', color='darkgreen', ax=ax1)
    ax1 = sns.lineplot(x='thres', y='pat_recall', data = df,
            label='recall', color='lightgreen', ax=ax1) #tab:green
    ax1 = sns.lineplot(x='thres', y='physionet2019_utility', data = df,
            label='utility', color='black', ax=ax1) 

    ax2 = ax1.twinx()
    ax2.set_ylabel(f'Earliness: {earliness_stat} #hours before onset', fontsize=16, color='red')
    ax2 = sns.lineplot(x='thres', y=earliness, data = df,
            label='earliness', color='red', ax=ax2)

    plt.xticks(np.arange(df['thres'].min(), df['thres'].max(), step=0.05))
    ax1.set_yticks(np.arange(0,1, step=0.1))

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0.05,0.85))
    ax1.get_legend().remove()


def plot_scores(df, ax, **kwargs):
    """Plot prediction scores."""
    ax.set_title('Prediction scores')
    ax = sns.histplot(df['thres'], ax=ax, **kwargs)


def plot_info(df, ax, recall_threshold=0.90):
    """Plot information textbox."""
    # Get precision at the pre-defined recall threshold (typically 90%,
    # but other values might be interesting).
    greater = df['pat_recall'] > recall_threshold
    index = df['pat_recall'][greater].argmin()
    info = df.loc[index]

    textbox = '\n'.join((
        f'Patient-based Recall:    {info["pat_recall"]:5.2f}',
        f'Patient-based Precision: {info["pat_precision"]:5.2f}',
        f'Threshold:               {info["thres"]:5.4f}',
        f'Earliness Median:        {info["earliness_median"]:5.2f}',
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
    parser.add_argument('--output_path', default='results/evaluation/plots')
    parser.add_argument('--earliness-stat', default='mean')

    args = parser.parse_args()
    input_path = args.input_path
    with open(input_path, 'r') as f:
        d = json.load(f)
    df = pd.DataFrame(d)

    earliness_stat = args.earliness_stat
    earliness = f'earliness_{earliness_stat}'

    fig, axes = plt.subplots(nrows=3, figsize=(10, 6), squeeze=True)

    plot_curves(df, axes[0])
    plot_scores(df, axes[1])
    plot_info(df, axes[2])

    out_file = os.path.split(input_path)[-1].split('.')[0] + '_' + earliness + '.png' 
    plt.savefig( os.path.join(args.output_path, out_file))
    plt.show()
