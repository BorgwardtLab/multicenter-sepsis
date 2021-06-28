"""Ranking and visualising of Shapley values over different runs."""

import argparse
import collections
import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from shap.plots.colors import blue_rgb

from src.torch.shap_utils import get_pooled_shapley_values


def make_plot(df, max_values=10):
    """Create a bar plot from a set of Shapley values."""
    fig, ax = plt.subplots()

    df = df.fillna(0)
    df = df.sort_values(by='mean', ascending=False)
    df = df.iloc[:max_values]

    ytick = range(max_values)

    # Following the look and feel of the 'original' Shap bar plot by
    # imitating its decorations.

    ax.set_yticks(ytick)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    print(df)

    ax.barh(
        ytick,
        df['mean'],
        xerr=df['sdev'],
        align='center',
        color=blue_rgb,
        edgecolor=(1, 1, 1, 0.8),
    )

    plt.xlabel('Mean absolute Shapley value')
    plt.tight_layout()
    plt.savefig('/tmp/shapley_bar.png')


def calculate_mean_with_sdev(
    data_frames,
    name,
    prefix,
    rank=True,
    collate=False,
):
    """Calculate features/ranks from data frames and store them in a file."""
    # Pretty ugly, but it does the trick :)
    data_frames = copy.copy(data_frames)

    # First, let's calculate ranks over each of the data frames
    # containing a number of Shapley values.
    for index, df in enumerate(data_frames):
        df = df.abs()
        df = df.mean(axis='rows')
        df = df.sort_values(ascending=False)
        df.index.name = 'feature'

        if rank:
            df = df.rank(ascending=False, method='min')
            df.name = 'rank'

        df = df.sort_index(kind='mergesort')

        data_frames[index] = df

    # Now let's concatenate all these data frames and calculate a mean
    # and a standard deviation over their ranks.

    df = pd.concat(data_frames, axis='columns')

    # If desired, collate mean absolute Shapley value over *variables*
    # instead of features.
    if collate:
        def feature_to_var(column):
            """Rename feature name to variable name."""
            column = column.replace('_count', '')
            column = column.replace('_raw', '')
            column = column.replace('_indicator', '')
            return column

        df = df.rename(feature_to_var, axis=0)
        df = df.groupby(level=0, axis=0).max()

    mean = df.mean(axis='columns')
    sdev = df.std(axis='columns')

    df['mean'] = mean
    df['sdev'] = sdev

    df = df[['mean', 'sdev']]

    if rank:
        df.to_csv(
            f'/tmp/shapley_{prefix}ranking_with_sdev_{name}.csv',
            na_rep='0.0',
            index=True
        )
    else:
        df.to_csv(
            f'/tmp/shapley_{prefix}mean_with_sdev_{name}.csv',
            na_rep='0.0',
            index=True
        )

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'FILE',
        nargs='+',
        type=str,
        help='Input file(s)`. Must contain Shapley values.',
    )

    parser.add_argument(
        '-i', '--ignore-indicators-and-counts',
        action='store_true',
        help='If set, ignores indicator and count features.'
    )

    parser.add_argument(
        '-H', '--hours-before',
        type=int,
        help='Uses only values of at most `H` hours before the maximum '
             'prediction score.'
    )

    parser.add_argument(
        '-l', '--last',
        action='store_true',
        help='If set, uses only the last value, i.e. the one corresponding '
             'to the maximum prediction score. This is equivalent to the   '
             'setting of `-H 1`.'
    )

    args = parser.parse_args()

    if args.last:
        args.hours_before = 1

    prefix = ''

    # Ensure that we track how we generated the rankings in case we
    # shift our predictions.
    if args.hours_before is not None:
        prefix += f'{args.hours_before}h_'

    # Will store all Shapley values corresponding to a single data set.
    # The idea behind this is to collect Shapley values from different
    # repetitions on the same data.
    dataset_to_shapley = collections.defaultdict(list)

    for filename in args.FILE:
        shap_values, feature_values, feature_names, dataset_name = \
            get_pooled_shapley_values(
                filename,
                args.ignore_indicators_and_counts,
                args.hours_before
            )

        dataset_to_shapley[dataset_name].append(
            (shap_values, feature_values)
        )

    # Having collected Shapley values for each data set, we may now
    # perform an 'inner collation', i.e. we calculate mean and standard
    # deviation of ranks for each data set indivudally..
    for dataset_name in dataset_to_shapley:
        values = dataset_to_shapley[dataset_name]

        data_frames = [
            pd.DataFrame(s, columns=feature_names) for s, _ in values
        ]

        calculate_mean_with_sdev(data_frames, dataset_name, prefix)
        df = calculate_mean_with_sdev(
            data_frames,
            dataset_name,
            prefix,
            rank=False
        )

    # Prepare ranking and calculations *between* data sets, thus
    # creating an 'external collation', i.e. the standard deviation
    # refers to the variability across data sets.
    dataset_to_ranks = {}

    # 'Flatten' Shapley values so that we only have one set of them per
    # data set.
    for dataset_name in dataset_to_shapley:
        values = dataset_to_shapley[dataset_name]

        s = np.vstack([s for s, _ in values])
        f = np.vstack([f for _, f in values])

        # Replace the previously-stored data sets with a single set of
        # Shapley values and features. This will enable us to collate,
        # albeit it over *variables* and not over features.
        dataset_to_shapley[dataset_name] = (s, f)

    # Pool Shapley values across all data sets. This permits an analysis
    # of the overall ranking.
    data_frames = [
        pd.DataFrame(s, columns=feature_names)
        for s, _ in dataset_to_shapley.values()
    ]

    df = calculate_mean_with_sdev(
        data_frames,
        'pooled',
        prefix,
        rank=False,     # Don't need the rank if we want to draw a bar plot
        collate=True,   # Collate over variables
    )

    make_plot(df)
