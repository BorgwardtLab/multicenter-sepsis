"""
Script to load trained estimator to evaluate it on another dataset.
"""

import argparse
from functools import partial
import json
import os
from time import time
import numpy as np
import pandas as pd
from hashlib import md5
import joblib
import pathlib
from sklearn.pipeline import Pipeline
from sklearn.metrics import SCORERS
from sklearn.metrics._scorer import _cached_call
from sklearn.model_selection import RandomizedSearchCV
#custom modules
from src.sklearn.data.utils import load_pickle, save_pickle
#from src.sklearn.loading import load_and_transform_data
from src.sklearn.main import load_data_splits
from src.evaluation import (
    get_physionet2019_scorer, StratifiedPatientKFold, shift_onset_label)
from src.sklearn.main import handle_label_shift
from src.variables.mapping import VariableMapping

VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)

def load_model(path):
    """ function to load model via joblib and compute checksum"""
    checksum = md5(open(path,'rb').read()).hexdigest() 
    with open(path, 'rb') as f:
        model = joblib.load(f)
    return model, checksum

def main():
    """Parse arguments and launch fitting of model."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_path', default='datasets/{}/data/parquet/features_middle',
        help='Path to input data, (dataset name not formatted, but left as {})'
    )
    parser.add_argument(
        '--model_path', default='results/hypersearch10_regression',
        help='Relative path to experimental results including trained models'
    )
    parser.add_argument(
        '--output_path', default=None,
        help='path to evaluation result file'
    )
    parser.add_argument(
        '--train_dataset', default='mimic',
        help='Train Dataset Name: [physionet2019, ..]'
    )
    parser.add_argument(
        '--eval_dataset', default='mimic_demo',
        help='Evaluation Dataset Name: [physionet2019, ..]'
    )
    #parser.add_argument(
    #    '--label-propagation', default=6, type=int,
    #    help='By how many hours to shift label into the past. Default: 6'
    #)
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
        '--split', default='validation', 
        help='on which split to evaluate [validation (default), test]'
    )
    parser.add_argument(
        '--variable_set', default='full',
        help="""which variable set should be used: [full, physionet], 
            where physionet refers to the subset as derived from 
            physionet challenge variables"""
    )
    parser.add_argument(
        '--feature_set', default='large',
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
        '--repetition_model', 
        help='bool if repetition model is used, or not', action='store_true', 
        default=False)
    parser.add_argument(
        '--task', default='classification', 
        help='which prediction task to use: [classification, regression]'
    )   
    parser.add_argument(
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )
    parser.add_argument(
        '--cost', default=0,
        type=int,
        help='lambda cost to use (default 0, inactive)'
    )


    args = parser.parse_args()
    method = args.method
    train_dataset = args.train_dataset
    eval_dataset = args.eval_dataset
    task = args.task
    rep = args.rep
    # pass this argument to data loading and pipeline creator 
    if args.method in ['sofa', 'qsofa', 'sirs', 'news', 'mews']:
        args.baselines = True
    else: 
        args.baselines = False

    args.dataset = eval_dataset #load_data_splits expect this property
    data, lam = load_data_splits(args, splits=[args.split])

    # label_shift function assumes a dataset arg:
    if task == 'classification':
        data = handle_label_shift(args, data)
 
    # Load pretrained model
    model_path = os.path.join(args.model_path, train_dataset + '_' + method)
    model_file = 'best_estimator' if not args.repetition_model else f'model_repetition_{rep}'
    model_file += '.pkl'
    model_path = os.path.join(model_path, model_file)
    print(f'Loading model from {model_path}')
    model, checksum = load_model(model_path) 

    # Select split for evaluation:
    split = args.split
    if split not in ['validation', 'test']:
        raise ValueError(f'{split} not among the valid eval splits: [validation, test]')
    X_eval = data[f'X_{split}']
    y_eval = data[f'y_{split}']
    tp_labels = data[f'tp_labels_{split}'] #only for down-stream eval 
    results = {}
    results['model'] = method
    results['model_path'] = model_path
    results['model_checksum'] = checksum
    results['model_params'] = model.get_params()
    results['dataset_train'] = train_dataset
    results['dataset_eval'] = eval_dataset
    results['split'] = split
    results['rep'] = rep 
    results['task'] = task
    results['feature_set'] = args.feature_set
    results['variable_set'] = args.variable_set 
 
    #results['predictions']
    if task == 'classification':
        preds = model.predict(X_eval)
        preds = pd.DataFrame(preds, index=y_eval.index)
         
        probas = model.predict_proba(X_eval)[:,1]
        probas = pd.DataFrame(probas, index=y_eval.index)
    else:   
        # predictions of regression model are not really probas, but we stay consistent with naming
        probas = model.predict(X_eval)
        probas = pd.DataFrame(probas, index=y_eval.index)
        preds = (probas > 0).astype(int) 

    ids = y_eval.index.get_level_values(VM_DEFAULT('id')).unique().tolist() 
    results['ids'] = ids
    labels = [] #unshifted, timepoint-wise labels
    targets = [] # target that was used in training: shifted label or regr. target
    predictions = []
    scores = []
    times = []

    for pid in ids:
        labels.append(tp_labels[pid].values.tolist())
        targets.append(y_eval[pid].values.tolist()) 
        predictions.append(preds.loc[pid][0].tolist())
        scores.append(probas.loc[pid][0].values.tolist()) 
        times.append(y_eval[pid].index.tolist())
    results['labels'] = labels
    results['targets'] = targets
    results['predictions'] = predictions
    results['scores'] = scores 
    results['times'] = times
    results['label_propagation'] = args.label_propagation #was only applied here for classification 
    
    outfile = args.output_path
    #os.makedirs(output_path, exist_ok=True) 
    #outfile = os.path.join(output_path, f'{method}_{train_dataset}_{eval_dataset}.json')

    #clf obj don't go into json format, remove them:
    for key in ['steps', 'est']:
        results['model_params'].pop(key, None)

    os.makedirs(os.path.split(outfile)[0], exist_ok=True) 
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ in "__main__":
    main()

