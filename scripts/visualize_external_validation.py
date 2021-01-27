"""Script for the visualization of external validation results."""
import argparse
from glob import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# These columns are not required for the visualization we are doing in this
# script
DROP_COLUMNS = ['model_params', 'labels', 'scores', 'predictions', 'ids']

METRICS = ['auroc', 'average_precision',
           'balanced_accuracy', 'physionet2019_score']


def load_json(filename):
    with open(filename, 'r') as f:
        d = json.load(f)
    # Remove some of the keys
    return {key: val for key, val in d.items() if key not in DROP_COLUMNS}


def format_dataset(dataset_name):
    prefixes = ['Preprocessed']
    suffixes = ['Dataset', '2019']
    for prefix in prefixes:
        if dataset_name.startswith(prefix):
            dataset_name = dataset_name[len(prefix):]
    for suffix in suffixes:
        if dataset_name.endswith(suffix):
            dataset_name = dataset_name[:-len(suffix)]
    return dataset_name


def main(search_folder, normalize, output_file):
    results = pd.DataFrame.from_records([
        load_json(f)
        for f in glob(
            os.path.join(search_folder, '**', '*.json'),
            recursive=True
        )
    ])
    results['dataset_train'] = results['dataset_train'].apply(format_dataset)
    results['dataset_eval'] = results['dataset_eval'].apply(format_dataset)

    if normalize:
        for dataset in results['dataset_train'].unique():
            self_performance = results.query(
                'dataset_train == @dataset & dataset_eval == @dataset').iloc[0][METRICS].values
            evaluated_on = results['dataset_eval'] == dataset
            normalized_metrics = (results[evaluated_on][METRICS] /
                                  self_performance).astype('float')
            results.loc[evaluated_on, METRICS] = normalized_metrics

    fig, axs = plt.subplots(1, len(METRICS), figsize=(6*len(METRICS), 5))
    for ax, metric in zip(axs, METRICS):
        pivot = results.pivot(
            index='dataset_train', columns='dataset_eval', values=metric)
        pivot = pivot.loc[pivot.index.sort_values()]
        pivot = pivot[pivot.columns.sort_values()]
        if normalize:
            sns.heatmap(pivot, annot=True, fmt='.3f',
                        square=True, ax=ax, cmap='RdBu', center=1.)
            ax.set_title(metric + ' scaled')
        else:
            sns.heatmap(pivot, annot=True, fmt='.3f', square=True, ax=ax)
            ax.set_title(metric)
    fig.savefig(output_file, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('search_folder')
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--output', required=True, type=str)
    args = parser.parse_args()

    main(args.search_folder, args.normalize, args.output)
