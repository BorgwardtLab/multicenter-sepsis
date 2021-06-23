"""Ranking of Shapley values over different runs."""

import argparse
import collections

import numpy as np
import pandas as pd

from src.torch.shap_utils import get_pooled_shapley_values


def calculate_ranks(shapley_values, feature_names, name):
    """Calculate ranks of features and store them in a file."""
    df = pd.DataFrame(shapley_values, columns=feature_names)
    df = df.abs()
    df = df.mean(axis='rows')
    df = df.sort_values(ascending=False)

    df.index.name = 'feature'
    df.name = 'mean'
    df.to_csv(f'/tmp/shapley_mean_{name}.csv', index=True)

    df = df.rank(ascending=False, method='min')
    df.name = 'rank'
    df.to_csv(f'/tmp/shapley_ranking_{name}.csv', index=True)

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

    # Will store all Shapley values corresponding to a single data set.
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

    dataset_to_ranks = {}

    # 'Flatten' Shapley values so that we only have one set of them per
    # data set.
    for dataset_name in dataset_to_shapley:
        values = dataset_to_shapley[dataset_name]

        s = np.vstack([s for s, _ in values])
        f = np.vstack([f for _, f in values])

        dataset_to_shapley[dataset_name] = (s, f)
        ranks = calculate_ranks(s, feature_names, dataset_name)

        # Store ranks so that we can calculate mean ranks across
        # different data sets.
        dataset_to_ranks[dataset_name] = ranks

    df_mean_rank = pd.concat(
        [ranks for ranks in dataset_to_ranks.values()],
        axis='columns',
    ).mean(axis='columns')

    df_mean_rank.to_csv('/tmp/shapley_mean_ranking.csv', index=True)

    # Pool Shapley values across all data sets. This permits an analysis
    # of the overall ranking.
    all_shap_values = np.vstack([
        np.vstack(
            [s for s, _ in values for values in dataset_to_shapley.items()]
        )
    ])

    calculate_ranks(all_shap_values, feature_names, 'pooled')

    raise 'heck'

    prefix = ''

    # Ensure that we track how we generated the plots in case we shift
    # our predictions.
    if args.hours_before is not None:
        prefix += f'{args.hours_before}h_'

    df = pd.DataFrame(all_shap_values, columns=feature_names)
    df = df.abs()
    df = df.mean(axis='rows')
    df = df.sort_values(ascending=False)

    print(df)