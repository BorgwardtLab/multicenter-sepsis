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

from sklearn.metrics import auc


def plot_curve(data_frames, ax, curve_type='roc'):
    """Plot evaluation curve of a certain type.""" 
    columns = ['pat_specificity', 'pat_recall']

    if curve_type == 'pr':
        columns = ['pat_recall', 'pat_precision']

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
        ax=ax,
    )

    if curve_type == 'roc':
        g.set_title('ROC curve')
        g.set_xlabel('FPR')
        g.set_ylabel('TPR')
        g.legend(loc='lower right')

        g.plot([0, 1], [0, 1], color='k', linestyle='dashed')

    elif curve_type == 'pr':
        g.set_title('PR curve')
        g.set_xlabel('Precision')
        g.set_ylabel('Recall')
        g.legend(loc='lower left')

        # TODO: get prevalence from somewhere and plot it?

    g.set_xlim((-0.01, 1.05))
    g.set_ylim((-0.01, 1.05))
    g.set_aspect(1.0)

    # Calculate AUCs for each model and update the legend accordingly.
    # Notice that this might be slightly less precise than calling the
    # `average_precision_score` or `roc_auc_score` functions, but here
    # we do not need the predictions, making the script easier to call
    # from jobs.
    aucs = [
        auc(df_.iloc[:, 0], df_.iloc[:, 1]) for _, df_ in df.groupby('model')
    ]

    for text, area in zip(g.legend_.texts, aucs):
        text.set_text(text.get_text() + f' (AUC: {area:.2f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'FILE',
        nargs='+',
        type=str,
        help='Input file(s). Must be created by the evaluation script.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file'
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

    fig, ax = plt.subplots(nrows=2, figsize=(4, 8))

    plot_curve(data_frames, ax[0])
    plot_curve(data_frames, ax[1], curve_type='pr')

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)

    if args.show:
        plt.show()