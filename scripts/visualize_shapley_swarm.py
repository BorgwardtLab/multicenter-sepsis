"""Visualisation of Shapley values as a swarm plot."""

import argparse
import os
import shap

import numpy as np

from src.torch.shap_utils import feature_to_name
from src.torch.shap_utils import get_pooled_shapley_values

import matplotlib

matplotlib.use('pgf')
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'sans-serif',
    'font.size': 16,
    'text.usetex': True,
    'pgf.rcfonts': True,
})

import matplotlib.pyplot as plt


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


def make_plots(
    shap_values,
    feature_values,
    feature_names,
    dataset_name,
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
        plt.savefig(filename_prefix + f'_{plot}.pgf', dpi=300)
        plt.savefig(filename_prefix + f'_{plot}.png', dpi=300)
        plt.clf()

    variables = [('Heart rate (raw)', 'hr'),
                 ('Mean arterial pressure (raw)', 'map'),
                 ('Temperature (raw)', 'temp')]

    if args.ignore_indicators_and_counts:
        variables = [('Heart rate', 'hr'),
                     ('Mean arterial pressure', 'map'),
                     ('Temperature', 'temp')]

    for variable, abbr in variables:
        index = feature_names.index(variable)

        shapleys = make_explanation(
            shap_values[:, index],
            feature_values[:, index],
            feature_names[index]
        )

        shap.plots.scatter(shapleys, hist=False)

        plt.gcf().set_size_inches(4, 2)

        plt.tight_layout()
        plt.savefig(filename_prefix + f'_scatter_{abbr}.pgf', dpi=300)
        plt.savefig(filename_prefix + f'_scatter_{abbr}.png', dpi=300)
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

    parser.add_argument(
        '--label',
        default=None,
        type=int,
        choices=[0, 1],
        help='If set, restrict analysis to a specific label, permitting '
             'individual plots for cases and controls, respectively.'
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
                args.hours_before,
                # Since we want to use scatter plots, we should use the
                # 'raw' or 'original' feature scales. Else, the scales
                # of the plots will look weird.
                return_normalised_features=False,
                label=args.label,
            )

        # If we ignore *all* categories anyway, there's no need to spell
        # them out when mapping features to their names.
        ignore_category = args.ignore_indicators_and_counts

        feature_names = list(map(
            lambda x: feature_to_name(x, ignore_category=ignore_category),
            feature_names)
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

    # Ensure that we track how we generated the plots in case we shift
    # our predictions.
    if args.hours_before is not None:
        prefix += f'{args.hours_before}h_'

    # Ditto for dropped indicators and count variables.
    if args.ignore_indicators_and_counts:
        prefix += 'raw_'

    # Ditto for label selection
    if args.label is not None:
        if args.label == 1:
            prefix += 'cases_'
        else:
            prefix += 'controls_'

    make_plots(
        all_shap_values,
        all_feature_values,
        feature_names,
        dataset_name,
        prefix=prefix,
    )