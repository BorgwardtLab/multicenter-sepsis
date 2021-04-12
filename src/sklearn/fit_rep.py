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

def load_data(args, split):
    """
    util function to load current data split
    """
    dataset = args.dataset
    rep = args.rep
    input_path = args.input_path.format(dataset)
    normalizer_path = os.path.join(args.normalizer_path, 
        f'normalizer_{dataset}_rep_{rep}.json' )
    if args.cost > 0:
        lam_file = f'lambda_{dataset}_rep_{rep}_cost_{args.cost}.json'
    else:
        lam_file = f'lambda_{dataset}_rep_{rep}.json'
    lambda_path = os.path.join(args.lambda_path, 
        lam_file )
    split_path = os.path.join(args.split_path, 
        f'splits_{dataset}.json' ) 

    # Load data and apply on-the-fly transforms:
    return load_and_transform_data(
        input_path,
        split_path,
        normalizer_path,
        lambda_path,
        args.feature_path,
        split=split,
        rep=rep,
        feature_set=args.feature_set,
        variable_set=args.variable_set,
        task=args.task,
        baselines=False
    )

def load_data_splits(args, 
    splits=['train','validation','test']):
    """
    Util function to read all 3 data splits of current repetion
    and return dictionary {X_train: [], y_train: [], ..}
    setting the label to the current task
    Parameters: 
    args: config object with properties:
        .task regression or classification
        .index: multi oder single index
        .input_path: path to input data, with dataset name generic with {}
        .dataset: name of dataset
        .split_path: path to split info
        .lambda_path: path to lambdas
        .normalizer_path: path to normalizer
        .features_path: path to feature columns file
        .variable_set: full, physionet
        .feature_set: large, small
        .rep: repetition/fold of split
    """ 
    d = {}
    if args.task == 'classification':
        label = VM_DEFAULT('label')
    elif args.task == 'regression':
        label = VM_DEFAULT('utility')
    else:
        raise ValueError(f'Task {args.task} not among valid tasks. ')
    for split in splits:
        data, lam = load_data(args, split)
        if args.index == 'multi':   
            data = data.reset_index().set_index(
                [VM_DEFAULT('id'), VM_DEFAULT('time')]
            ) 
        d[f'y_{split}'] = data[label]
        # shifted and unshifted labels for down stream eval irrespecive of task:
        d[f'tp_labels_{split}'] = data[VM_DEFAULT('label')]
        d[f'tp_labels_shifted_{split}'] = data[VM_DEFAULT('label')]
        data = data.drop(columns=[ 
            VM_DEFAULT('label'), VM_DEFAULT('utility')
            ], errors='ignore'
        )
        #if args.task == 'regression':
        #    data = data.drop(columns=[label])
        # sanity check as we must not leak any label info to the input data
        assert all( 
            [ VM_DEFAULT(x) not in data.columns 
                for x in ['label', 'utility'] 
            ]
        )
        d[f'X_{split}'] = data 
    return d, lam


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
