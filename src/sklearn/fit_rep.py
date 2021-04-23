"""Refit sklearn pipeline (with previously searched hyperparameters) on repetition fold of dataset"""
import argparse
from functools import partial
import json
import os
import pathlib
from time import time
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import SCORERS
from sklearn.metrics._scorer import _cached_call
from sklearn.model_selection import RandomizedSearchCV

from src.main import load_data_splits
from src.variables.mapping import VariableMapping
from src.sklearn.data.utils import load_pickle, save_pickle
from src.sklearn.loading import load_and_transform_data
from src.sklearn.shift_utils import handle_label_shift
from src.evaluation import (
    get_physionet2019_scorer, StratifiedPatientKFold)

VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)

def main():
    """Parse arguments and launch fitting of model."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_path', default='datasets/{}/data/parquet/features',
        help='Path to input data directory (relative from dataset directory)'
    )
    parser.add_argument(
        '--result_path', default='results',
        help='Relative path to experimental results (from input path)'
    )
    parser.add_argument(
        '--dataset', default='mimic_demo',
        help='Dataset Name: [physionet2019, ..]'
    )
    parser.add_argument(
        '--label_propagation', default=6, type=int,
        help="""(Active for classification task) By how many hours to 
            shift label into the past. Default: 6"""
    )
    parser.add_argument(
        '--label_propagation_right', default=24, type=int,
        help="""(Active for classification task) By how many hours to 
            shift label into the future, afterwards 0 again. Default: 24"""
    )
    parser.add_argument(
        '--method', default='lgbm', type=str,
        help='Method to use for classification [lgbm, lr]'
    )
    parser.add_argument(
        '--clf_params', nargs='+', default=[],
        help='Parameters passed to the classifier constructor'
    )
    parser.add_argument(
        '--cv_n_jobs', type=int, default=10,
        help='n_jobs for cross-validation'
    )
    parser.add_argument(
        '--variable_set', default='full',
        help="""which variable set should be used: [full, physionet], 
            where physionet refers to the subset as derived from 
            physionet challenge variables"""
    )
    parser.add_argument(
        '--feature_set', default='middle',
        help="""which feature set should be used: [large, small], 
            large including feature engineering for classic models"""
    )
    parser.add_argument(
        '--split_path', 
        help='path to split file', 
        default='config/splits'
    )
    parser.add_argument(    
        '--normalizer_path', 
        help='path to normalization stats', 
        default='config/normalizer'
    )
    parser.add_argument(
        '--lambda_path', 
        help='path to lambda files', 
        default='config/lambdas'
    )
    parser.add_argument(
        '--feature_path', 
        help='path to feature names file', 
        default='config/features.json'
    )
    parser.add_argument(
        '--rep', 
        help='split repetition', type=int, 
        default=0)
    parser.add_argument(
        '--task', default='regression', 
        help='which prediction task to use: [classification, regression]'
    )
    parser.add_argument(
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )
    parser.add_argument(
        '--cost', default=0,
        type=int,
        help='lambda cost to use (default 0 (inactive))'
    )
    parser.add_argument(
        '--target_name', default='neg_log_loss',
        help='Only for classification: which objective to optimize in model selection [physionet_utility, roc_auc, average_precision]'
    )


    args = parser.parse_args()
    ## Process arguments:
    task = args.task 
 
    # Load data and current lambda and apply on-the-fly transforms:
    data, lam = load_data_splits(args)
    #data = load_data_from_input_path(
    #    args.input_path, args.dataset, args.index, args.extended_features)
    if task == 'classification':
        # for regression task the label shift happens in target calculation
        data = handle_label_shift(args, data)
 
    # TODO: add (update) baseline option! 
    ## for baselines: 
    #if args.method in ['sofa', 'qsofa', 'sirs', 'news', 'mews']:
    #    # use baselines as prediction input data
    #    # hack until prepro is run again (currently live jobs depending on it)
    #    data['baselines_train'].index = data['X_train'].index
    #    data['baselines_validation'].index = data['X_validation'].index
    #    data['X_train'] = data['baselines_train']
    #    data['X_validation'] = data['baselines_validation']

    # TODO: Load pretrained best estimator model:
    result_path = os.path.join(args.result_path, args.dataset+'_'+args.method)
    checkpoint_path = os.path.join(result_path, 'best_estimator.pkl')
    pipe = joblib.load(checkpoint_path)
    # sanity check that no model has warm_start=True as this would mess with refitting
    params = pipe['est'].get_params()
    if 'warm_start' in params.keys():
        assert not params['warm_start']

    # Fit on current repetition split:
    pipe.fit(data['X_train'], data['y_train'])
    # Dump estimator:
    joblib.dump(
        pipe,
        os.path.join(result_path, f'model_repetition_{args.rep}.pkl'),
        compress=1
    )

if __name__ in "__main__":
    main()
