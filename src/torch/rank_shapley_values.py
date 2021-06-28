"""Ranking of Shapley values over different runs."""

import argparse
import collections
import copy

import numpy as np
import pandas as pd

from src.torch.shap_utils import get_pooled_shapley_values


def calculate_ranks(shapley_values, feature_names, name, prefix, store=True):
    """Calculate ranks of features and store them in a file."""
    df = pd.DataFrame(shapley_values, columns=feature_names)
    df = df.abs()
    df = df.mean(axis='rows')
    df = df.sort_values(ascending=False)

    df.index.name = 'feature'
    df.name = 'mean'

    if store:
        df.to_csv(f'/tmp/shapley_{prefix}mean_{name}.csv', index=True)

    df = df.rank(ascending=False, method='min')
    df.name = 'rank'

    if store:
        df.to_csv(f'/tmp/shapley_{prefix}ranking_{name}.csv', index=True)

    return df


def calculate_mean_with_sdev(data_frames, name, prefix, rank=True):
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

    mean = df.mean(axis='columns')
    sdev = df.std(axis='columns')

    df['mean'] = mean
    df['sdev'] = sdev

    df = df[['mean', 'sdev']]

    if rank:
        df.to_csv(
            f'/tmp/shapley_{prefix}ranking_with_sdev_{name}.csv',
            index=True
        )
    else:
        df.to_csv(
            f'/tmp/shapley_{prefix}mean_with_sdev_{name}.csv',
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
        '-c', '--collapse-features',
        action='store_true',
        help='If set, use only variables, collapsing features according '
             'to the aggregation function.'
    )

    parser.add_argument(
        '-a', '--aggregation-function',
        choices=['absmax', 'max', 'mean', 'median', 'min'],
        default='absmax',
        help='Aggregation function to use when features are collapsed to '
             'variables.'
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

        # Ranks of features/variables for this particular data set.
        # Later on, we will calculate a mean and standard deviation.
        ranks = calculate_ranks(
            s,
            feature_names,
            dataset_name,
            prefix,
            store=False
        )

        # Store ranks so that we can calculate mean ranks across
        # different data sets.
        dataset_to_ranks[dataset_name] = ranks

    df_mean_rank = pd.concat(
        [ranks for ranks in dataset_to_ranks.values()],
        axis='columns',
    ).mean(axis='columns')

    df_mean_rank.name = 'mean_rank'

    df_sdev_rank = pd.concat(
        [ranks for ranks in dataset_to_ranks.values()],
        axis='columns',
    ).std(axis='columns')

    df_sdev_rank.name = 'sdev_rank'

    df = pd.concat([df_mean_rank, df_sdev_rank], axis=1)
    df = df.fillna(0)
    df.to_csv(f'/tmp/shapley_{prefix}mean_ranking.csv', index=True)

    # Pool Shapley values across all data sets. This permits an analysis
    # of the overall ranking.
    all_shap_values = np.vstack([
        np.vstack(
            [s for s, _ in values for values in dataset_to_shapley.items()]
        )
    ])

    calculate_ranks(all_shap_values, feature_names, 'pooled', prefix)
