"""Visualisation of global Shapley values."""

import argparse
import pickle
import shap

import numpy as np
import matplotlib.pyplot as plt


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
        data = pickle.load(f)

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

    # Tensor has shape `n_samples, max_length, n_features`, where
    # `max_length` denotes the maximum length of a time series in
    # the batch.
    shap_values = data['shap_values']
    shap_values = shap_values[:, :, important_indices]

    # We need to pool all valid Shapley values over all time steps,
    # pretending that they are individual samples.
    #
    # This is the straightforward way of doing it. Don't judge.

    shap_values_pooled = []

    for index, values in enumerate(shap_values):
        length = lengths[index]
        values = values[:length, :]

        values = values[~np.isnan(values).all(axis=1), :]
        shap_values_pooled.append(values)

    shap_values_pooled = np.vstack(shap_values_pooled)

    shap.summary_plot(
        shap_values_pooled,
        feature_names=selected_features,
        plot_type='dot',
        show=False,
    )
    plt.tight_layout()
    plt.savefig('/tmp/shap_dot.png')

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
