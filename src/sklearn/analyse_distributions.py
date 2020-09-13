"""Analyse distributions of train--test splits"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.sklearn.data.utils import load_data

matplotlib.use('agg')


def analyse_feature_distributions(
    dataset_name,
    X_train, X_val, X_test,
):
    """Analyse feature distributions for a given data set."""
    features = [c for c in X_train.columns if 'hours' not in c]

    data_frames = [
        ('train', X_train),
        ('val', X_val),
        ('test', X_test)
    ]

    for feature in features:

        # Ensures that we do not run into any problems when storing
        # features in files.
        feature = feature.replace('/', '_')

        fig, ax = plt.subplots(nrows=2, ncols=3, squeeze=False)
        fig.suptitle(feature)

        for index, (name, df) in enumerate(data_frames):
            sns.histplot(
                df[feature],
                bins=50,
                kde=True,
                ax=ax[0, index]
            )

            ax[0, index].set_title(name)
            ax[1, index].set_aspect(4.0)

            sns.boxplot(
                x=df[feature],
                ax=ax[1, index]
            )

        print(f'Storing distributions for {feature}...')

        out_name = f'{dataset_name}_{feature}.png'

        plt.tight_layout()

        plt.savefig(
            os.path.join('/tmp', out_name),
            dpi=300,
        )

        plt.close()


def load_data_from_input_path(input_path, dataset_name, index='multi'):
    """Load the data according to dataset_name, and index-handling.

    Returns:
        Tuple of X_train, X_validation, X_test, y_train, y_validation,
        and y_test
    """
    input_path = os.path.join(
        input_path,
        'datasets',
        dataset_name,
        'data',
        'sklearn'
    )

    data = load_data(
        path=os.path.join(input_path, 'processed'),
        index=index,
        load_test_split=True,
    )

    return (
        data['X_train'],
        data['X_validation'],
        data['X_test'],
        data['y_train'],
        data['y_validation'],
        data['y_test']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        default='mimic3',
        type=str,
        help='Dataset Name: [mimic3, ..]'
    )

    parser.add_argument(
        '-p', '--path',
        default='/links/groups/borgwardt/Projects/sepsis/multicenter-sepsis/',
        type=str,
        help='Input path for data sets'
    )

    args = parser.parse_args()

    X_train, X_val, X_test, \
        y_train, y_val, y_test = load_data_from_input_path(
            args.path,
            args.dataset,
        )

    # Just a little bit of insurance: we should make sure that we are
    # working with the same feature sets.
    assert (X_train.columns == X_val.columns).all()
    assert (X_train.columns == X_test.columns).all()

    analyse_feature_distributions(
        args.dataset,
        X_train, X_val, X_test
    )
