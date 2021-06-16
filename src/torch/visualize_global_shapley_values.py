"""Visualisation of global Shapley values."""

import argparse
import io
import os
import pickle
import shap
import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.torch.shap_utils import get_model_and_dataset


class PickledTorch(pickle.Unpickler):
    """Unpickler for `torch` objects.

    Simple wrapper for `torch` objects that need to be serialised to
    a CPU instead of a GPU. Permits the analysis of such files, even
    if CUDA is not present.
    """

    def find_class(self, module, name):
        """Return class for handling functions of a certain type."""
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def get_run_id(filename):
    """Extract run ID from filename."""
    run_id = os.path.basename(filename)
    run_id = os.path.splitext(run_id)[0]
    run_id = run_id[:8]
    run_id = f'sepsis/mc-sepsis/runs/{run_id}'
    return run_id


def pool(lengths, shapley_values, feature_values):
    """Pool Shapley values and feature values.

    This function performs the pooling step required for Shapley values
    and feature values. Each time step of a time series will be treated
    as its own sample instance.

    Returns
    -------
    Tuple of pooled Shapely values and feature values.
    """
    # This is the straightforward way of doing it. Please don't judge,
    # or, if you do, don't judge too much :)
    shapley_values_pooled = []
    feature_values_pooled = []

    for index, (s_values, f_values) in enumerate(
        zip(shapley_values, feature_values)
    ):
        length = lengths[index]
        s_values = s_values[:length, :]
        f_values = f_values[:length, :]

        # Need to store mask for subsequent calculations. This is
        # required because we must only select features for which
        # the Shapley value is defined!
        mask = ~np.isnan(s_values).all(axis=1)

        s_values = s_values[mask, :]
        f_values = f_values[mask, :]

        shapley_values_pooled.append(s_values)
        feature_values_pooled.append(f_values)

    shapley_values_pooled = np.vstack(shapley_values_pooled)
    feature_values_pooled = np.vstack(feature_values_pooled)

    return shapley_values_pooled, feature_values_pooled


def get_aggregation_function(name):
    """Return aggregation function."""
    if name == 'absmax':
        def absmax(x):
            return max(x.min(), x.max(), key=abs)
        return absmax
    elif name == 'max':
        return np.max
    elif name == 'min':
        return np.min
    elif name == 'mean':
        return np.mean
    elif name == 'median':
        return np.median


def make_explanation(shapley_values, feature_values, feature_names):
    """Wrap Shapley values in an `Explanation` object."""
    return shap.Explanation(
        # TODO: does this base value make sense? We could always get the
        # model outputs by updating the analysis.
        base_values=0.0,
        values=shapley_values,
        data=feature_values,
        feature_names=feature_names,
    )


def process_file(filename, ignore_indicators_and_counts=False):
    """Process file.

    Parameters
    ----------
    filename : str
        Input filename.

    ignore_indicators_and_counts : bool, optional
        If set, ignores indicator and count features, thus reducing the
        number of features considered.

    Returns
    -------
    Tuple of pooled Shapley values, features, feature names, and data
    set name.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    _, _, out = get_model_and_dataset(
        get_run_id(filename), return_out=True
    )

    dataset_name = out['model_params']['dataset']
    model_name = out['model']

    assert model_name == 'AttentionModel', RuntimeError(
        'Shapley analysis is currently only supported for the '
        'AttentionModel class.'
    )

    feature_names = data['feature_names']
    lengths = data['lengths'].numpy()

    if args.ignore_indicators_and_counts:
        keep_features = [
            True if not col.endswith('indicator') and not col.endswith('count')
            else False
            for col in feature_names
        ]

        important_indices = np.where(keep_features)[0]
        selected_features = np.array(feature_names)[important_indices]
    else:
        important_indices = np.arange(0, len(feature_names))
        selected_features = feature_names

    # ignore positional encoding
    important_indices += 10

    # Tensor has shape `n_samples, max_length, n_features`, where
    # `max_length` denotes the maximum length of a time series in
    # the batch.
    shap_values = data['shap_values']
    shap_values = shap_values[:, :, important_indices]
    features = data['input'].numpy()
    features = features[:, :, important_indices]

    shap_values_pooled, features_pooled = pool(
        lengths,
        shap_values,
        features,
    )

    return (shap_values_pooled,
            features_pooled,
            selected_features,
            dataset_name)


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
        plt.cla()

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
            column = column.replace('_derived', '')
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
        plt.cla()


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
        default='max',
        help='Aggregation function to use when features are collapsed to '
             'variables.'
    )

    args = parser.parse_args()

    all_shap_values = []
    all_feature_values = []
    all_datasets = []

    for filename in args.FILE:
        shap_values, feature_values, feature_names, dataset_name = \
            process_file(
                filename, args.ignore_indicators_and_counts
            )

        all_shap_values.append(shap_values)
        all_feature_values.append(feature_values)
        all_datasets.append(dataset_name)

    all_shap_values = np.vstack(all_shap_values)
    all_feature_values = np.vstack(all_feature_values)

    print(f'Analysing Shapley values of shape {all_shap_values.shape}')

    assert len(np.unique(all_datasets)) == 1, RuntimeError(
        'Runs must not originate from different data sets.'
    )

    prefix = os.path.basename(args.FILE[0])
    prefix = prefix.split('_')[0] + '_'

    print(f'Collating runs with prefix = {prefix}...')

    make_plots(
        all_shap_values,
        all_feature_values,
        feature_names,
        dataset_name,
        prefix=prefix,
        collapse_features=args.collapse_features,
        aggregation_function=args.aggregation_function,
    )
