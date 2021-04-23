"""Plot set of model curves.

This script draws evaluation curves for all models, supporting multiple
inputs. Only evaluation files are required.

Example call:

    python -m scripts.plots.plot_model_curves \
        --output-directory /tmp/              \
        FILE [FILE ...]

This will create plots in `tmp`.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from src.evaluation.patient_evaluation import format_dataset


def plot_curve(data_frames, ax, curve_type='roc'):
    """Plot evaluation curve of a certain type.""" 
    columns = ['pat_specificity', 'pat_recall']

    if curve_type == 'pr':
        columns = ['pat_precision', 'pat_recall']

    data_frames = [
        df[columns + ['model']] for df in data_frames
    ]


    df = pd.concat(data_frames)

    # We could also rename the column but this is easier.
    if curve_type == 'roc':
        df['pat_specificity'] = 1 - df['pat_specificity']

    g = sns.lineplot(
        x=columns[0],
        y=columns[1],
        data=df,
        hue='model',
        ci=None,
        ax=ax
    )

    if curve_type == 'roc':
        g.set_title('ROC curve')
        g.set_xlabel('FPR')
        g.set_ylabel('TPR')

    g.set_xlim((-0.05, 1.05))
    g.set_ylim((-0.05, 1.05))
    g.set_aspect(1.0)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'FILE',
        nargs='+',
        type=str,
        help='Input file(s). Must be created by the evaluation script.'
    )
    parser.add_argument(
        '--output-directory',
        default='results/evaluation/plots',
        type=str,
        help='Output directory'
    )

    parser.add_argument(
        '-s', '--show',
        action='store_true',
        help='If set, indicates that the resulting plots should be shown, '
             'not only saved to a file.'
    )

    args = parser.parse_args()

    data_frames = []

    for filename in args.FILE:
        with open(filename) as f:

            # TODO: in the absence of any identifying information, let's
            # use the filename to show individual curves. This is just a 
            # simple proof of concept.
            df = pd.DataFrame(json.load(f))
            df['model'] = os.path.splitext(os.path.basename(filename))[0]

            data_frames.append(df)

    fig, ax = plt.subplots()

    plot_curve(data_frames, ax)

    # for setting title:
    xmin, xmax = plot_curves(df, axes[0], names,level=level)
    plot_scores(df, scores, axes[1])
    plot_proportion_curves(df, axes[2])
    info = plot_info(df, axes[3], level=level, recall_threshold=recall_thres)
    # vertical cut-off:
    for ax in axes[:3]:
        ax.axvline(info['thres'], color='black')
    # prevalence hline:
    if level == 'pat':
        prev = 'case_prevalence'
    elif level == 'tp':
        prev = 'tp_prevalence'
    axes[0].axhline(
        prev_dict[format_dataset(names['eval_dataset'])]['validation'][prev], 
        linestyle='--', color='black'
    )
 
    out_file = os.path.split(input_path)[-1].split('.')[0] + f'_{earliness}_{level}_thres_{100*recall_thres}.png' 
    plt.tight_layout()
    plt.savefig( os.path.join(args.output_path, out_file))

    if args.show:
        plt.show()
