"""Visualisation of global Shapley values."""

import argparse
import io
import pickle
import shap
import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        default='./dump/shapley/shap_test.pkl',
        type=str,
        help='Input file. Must contain Shapley values.',
    )

    parser.add_argument(
        '-i', '--ignore-indicators-and-counts',
        action='store_true',
        help='If set, ignores indicator and count features.'
    )

    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        data = PickledTorch(f).load()

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

    # We need to pool all valid Shapley values over all time steps,
    # pretending that they are individual samples.
    #
    # This is the straightforward way of doing it. Don't judge.

    shap_values_pooled = []
    features_pooled = []

    for index, (values, features_) in enumerate(zip(shap_values, features)):
        length = lengths[index]
        values = values[:length, :]
        features_ = features_[:length, :]

        # Need to store mask for subsequent calculations. This is
        # required because we must only select features for which
        # the Shapley value is defined.
        mask = ~np.isnan(values).all(axis=1)

        values = values[mask, :]
        features_ = features_[mask, :]

        shap_values_pooled.append(values)
        features_pooled.append(features_)

    shap_values_pooled = np.vstack(shap_values_pooled)
    features_pooled = np.vstack(features_pooled)

    # Optional filtering and merging over the columns, as specified by
    # the user. This permits us to map features to their corresponding
    # variables.
    
    # HIC SVNT LEONES
    #df = pd.DataFrame(shap_values_pooled, columns=selected_features)

    shap.summary_plot(
        shap_values_pooled,
        features=features_pooled,
        feature_names=selected_features,
        plot_type='dot',
        show=False,
    )
    plt.tight_layout()
    plt.savefig('/tmp/shap_dot.png')

    plt.show()
    plt.cla()

    shap.summary_plot(
        shap_values_pooled,
        feature_names=selected_features,
        plot_type='bar',
        show=False,
    )
    plt.tight_layout()
    plt.savefig('/tmp/shap_bar.png')

    plt.show()
