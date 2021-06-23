"""Ranking of Shapley values over different runs."""

import argparse
import os
import pickle
import shap

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.torch.shap_utils import get_pooled_shapley_values


def make_plots(
    shap_values,
    feature_values,
    feature_names,
    dataset_name,
    collapse_features=False,
    aggregation_function='max',
    prefix='',
    out_dir=None,
):
    """Create all possible plots.

    Parameters
    ----------
    shap_values : np.array of size (n, m)
        Shapley values to visualise.

    feature_values : np.array of size (n, m)
        The feature values corresponding to the Shapley values, i.e. the
        raw values giving rise to a certain Shapley value.

    feature_names : list of str
        Names to use for the features.

    dataset_name : str
        Name of the data set for which the visualisations are being
        prepared. Will be used to generate filenames.

    collapse_features : bool, optional
        If set, collapse features into their corresponding variables,
        and aggregate their respective values.

    aggregation_function : str, optional
        Sets aggregation function for feature collapse operation. Will
        only be used if `collapse_features == True`.

    prefix : str, optional
        If set, adds a prefix to all filenames. Typically, this prefix
        can come from the run that was used to create the Shapley
        values. Will be ignored if not set.

    out_dir : str or `None`
        Output directory for creating visualisations. If set to `None`,
        will default to temporary directory.
    """
    if out_dir is None:
        out_dir = '/tmp'

    filename_prefix = os.path.join(
        out_dir, 'shapley_' + prefix + dataset_name
    )

    plt.title(dataset_name)

    for plot in ['bar', 'dot']:
        shap.summary_plot(
            shap_values,
            features=feature_values,
            feature_names=feature_names,
            plot_type=plot,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(filename_prefix + f'_{plot}.png', dpi=300)
        plt.clf()

    # Optional filtering and merging over the columns, as specified by
    # the user. This permits us to map features to their corresponding
    # variables.
    if collapse_features:
        df = pd.DataFrame(shap_values, columns=feature_names)

        def feature_to_var(column):
            """Rename feature name to variable name."""
            column = column.replace('_count', '')
            column = column.replace('_raw', '')
            column = column.replace('_indicator', '')
            return column

        aggregation_fn = get_aggregation_function(aggregation_function)

        print(
            f'Collapsing features to variables using '
            f'{aggregation_function}...'
        )

        df = df.rename(feature_to_var, axis=1)
        df = df.groupby(level=0, axis=1).apply(
            lambda x: x.apply(aggregation_fn, axis=1)
        )

        shap_values = df.to_numpy()
        feature_names = df.columns

        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            plot_type='bar',
            show=False,
        )
        plt.tight_layout()
        plt.savefig(filename_prefix + '_variables_bar.png', dpi=300)
        plt.clf()


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

    all_shap_values = []
    all_feature_values = []
    all_datasets = []

    if args.last:
        args.hours_before = 1

    for filename in args.FILE:
        shap_values, feature_values, feature_names, dataset_name = \
            get_pooled_shapley_values(
                filename,
                args.ignore_indicators_and_counts,
                args.hours_before
            )

        all_shap_values.append(shap_values)
        all_feature_values.append(feature_values)
        all_datasets.append(dataset_name)

    all_shap_values = np.vstack(all_shap_values)
    all_feature_values = np.vstack(all_feature_values)

    print(f'Analysing Shapley values of shape {all_shap_values.shape}')

    prefix = ''

    # Ensure that we track how we generated the plots in case we shift
    # our predictions.
    if args.hours_before is not None:
        prefix += f'{args.hours_before}h_'

    df = pd.DataFrame(all_shap_values, columns=feature_names)
    print(df)
