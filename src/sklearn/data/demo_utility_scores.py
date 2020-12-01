import argparse
import os
import pickle
import sys

import pandas as pd

from sklearn.pipeline import Pipeline

from .transformers import *
from src.sklearn.data.utils import load_pickle, save_pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=str,
        help='Path from dataset to pickled df file to use as input',
        default='data/sklearn/processed'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset to use',
        default='demo'
    )
    parser.add_argument(
        '--split',
        type=str,
        help='Which split to use from [train, validation, test]; if not set '
             'we loop over all', 
    )

    args = parser.parse_args()
    split = args.split 
    dataset = args.dataset 
    path = os.path.join('datasets', dataset, args.path)
     
    if not split:
        splits = ['train', 'validation', 'test']
    else:
        splits = [split]

    data_pipeline = Pipeline([
        ('lookback_features', LookbackFeatures()),
        ('filter_invalid_times', InvalidTimesFiltration()),
        ('drop_cols', DropColumns(save=False)),
        ('calculate_utility_scores',
            CalculateUtilityScores(passthrough=False))]
    )

    print(f'Processing {dataset} and splits {splits}')

    for split in splits: 
        name = f'X_features_{split}'
        features_path = os.path.join(path, name + '.pkl')
        X = load_pickle(features_path)
        X = data_pipeline.fit_transform(X)

        print(f'Mean utility score for {split}:')
        print(X.groupby('id')['utility'].mean())
