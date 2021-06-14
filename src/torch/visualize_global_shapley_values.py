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


def pool(shapley_values, feature_values):
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
            max(x.min(), x.max(), key=abs)
        return absmax
    elif name == 'max':
        return np.max
    elif name == 'min':
        return np.min
    elif name == 'mean':
        return np.mean
    elif name == 'median':
        return np.median


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'FILE',
        type=str,
        help='Input file. Must contain Shapley values.',
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

    with open(args.FILE, 'rb') as f:
        data = pickle.load(f)

    _, _, out = get_model_and_dataset(
        get_run_id(args.FILE), return_out=True
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
        shap_values,
        features
    )

    shap.summary_plot(
        shap_values_pooled,
        features=features_pooled,
        feature_names=selected_features,
        plot_type='dot',
        show=False,
    )
    plt.tight_layout()
    plt.savefig('/tmp/shap_dot.png')

    plt.cla()

    # Optional filtering and merging over the columns, as specified by
    # the user. This permits us to map features to their corresponding
    # variables.
    if args.collapse_features:
        df = pd.DataFrame(shap_values_pooled, columns=selected_features)

        def feature_to_var(column):
            """Rename feature name to variable name."""
            column = column.replace('_count', '')
            column = column.replace('_raw', '')
            column = column.replace('_indicator', '')
            column = column.replace('_derived', '')
            return column

        aggregation_fn = get_aggregation_function(args.aggregation_function)

        print(
            f'Collapsing features to variables using '
            f'{args.aggregation_function}...'
        )

        df = df.rename(feature_to_var, axis=1)
        df = df.groupby(level=0, axis=1).apply(
            lambda x: x.apply(aggregation_fn, axis=1)
        )

        shap_values_pooled = df.to_numpy()
        selected_features = df.columns

    shap.summary_plot(
        shap_values_pooled,
        feature_names=selected_features,
        plot_type='bar',
        show=False,
    )
    plt.tight_layout()
    plt.savefig('/tmp/shap_bar.png')
