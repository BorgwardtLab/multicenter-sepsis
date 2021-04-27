"""Plot set of model scatterplots.

This script draws evaluation curves for all models, supporting multiple
inputs. Only evaluation files are required.

Example call:

    python -m scripts.plots.plot_model_curves \
        --output-directory /tmp/              \
        FILE

This will create plots in `tmp`.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from sklearn.metrics import auc


def interpolate_at(df, x):
    """Interpolate a data frame at certain positions.

    This is an auxiliary function for interpolating an indexed data
    frame at a certain position or at certain positions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame; must have index that is compatible with `x`.

    x : scalar or iterable
        Index value(s) to interpolate the data frame at. Must be
        compatible with the data type of the index.

    Returns
    -------
    Data frame evaluated at the specified index positions.
    """
    # Check whether object support iteration. If yes, we can build
    # a sequence index; if not, we have to convert the object into
    # something iterable.
    try:
        _ = (a for a in x)
        new_index = pd.Index(x)
    except TypeError:
        new_index = pd.Index([x])

    # Ensures that the data frame is sorted correctly based on its
    # index. We use `mergesort` in order to ensure stability. This
    # set of options will be reused later on.
    sort_options = {
        'ascending': False,
        'kind': 'mergesort',
    }
    df = df.sort_index(**sort_options)

    # TODO: have to decide whether to keep first index reaching the
    # desired level or last. The last has the advantage that it's a
    # more 'pessimistic' estimate since it will correspond to lower
    # thresholds.
    df = df[~df.index.duplicated(keep='last')]

    # Include the new index, sort again and then finally interpolate the
    # values.
    df = df.reindex(df.index.append(new_index).unique())
    df = df.sort_index(**sort_options)
    df = df.interpolate()

    return df.loc[new_index]


def get_coordinates(df, recall_threshold, level, x_stat='earliness_mean'):
    """Get coordinate from model-based data frame."""
    recall_col = f'{level}_recall'
    precision_col = f'{level}_precision'

    df = df.set_index(recall_col)
    df = interpolate_at(df, recall_threshold)

    assert len(df) == 1, RuntimeError(
        f'Expected a single row, got {len(df)}.'
    )

    x = df[x_stat].values[0]
    y = df[precision_col].values[0]

    # TODO: find nicer names for these labels
    return x, x_stat, y, precision_col


def make_scatterplot(df, ax, recall_threshold=0.50, level='pat'):

    plot_df = []

    for (model, repetition), df_ in df.groupby(['model', 'rep']):
        # We are tacitly assuming that all the labels remain the same
        # during the iteration.
        x, xlabel, y, ylabel = get_coordinates(df_, recall_threshold, level)

        plot_df.append(
            pd.DataFrame.from_dict({
                'x': [x],
                'y': [y],
                'model': [model],
                'repetition': [repetition],
            })
        )

    plot_df = pd.concat(plot_df)

    g = sns.scatterplot(
        x='x', y='y',
        data=plot_df,
        hue='model',
        ax=ax
    )

    g.set_ylabel(f'{ylabel} @ {recall_threshold:.2f}R')
    g.set_xlabel(xlabel)


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
        type=str,
        help='Input file. Must be a CSV containing information about all of '
             'the repetition runs of a model.'
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

    df = pd.read_csv(args.FILE)

    fig, ax = plt.subplots(nrows=2, figsize=(4, 8))

    make_scatterplot(df, ax[0])

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)

    if args.show:
        plt.show()
