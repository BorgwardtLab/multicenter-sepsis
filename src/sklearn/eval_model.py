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
        '--input_path', default='datasets/{}/data/parquet/features',
        help='Path to input data, (dataset name not formatted, but left as {})'
    )
    parser.add_argument(
        '--model_path', default='results/hypersearch10_regression',
        help='Relative path to experimental results including trained models'
    )
    parser.add_argument(
        '--output_path', default='results/evaluation',
        help='Relative path to evaluation results'
    )
    parser.add_argument(
        '--train_dataset', default='mimic',
        help='Train Dataset Name: [physionet2019, ..]'
    )
    parser.add_argument(
        '--eval_dataset', default='mimic_demo',
        help='Evaluation Dataset Name: [physionet2019, ..]'
    )
    parser.add_argument(
        '--label-propagation', default=6, type=int,
        help='By how many hours to shift label into the past. Default: 6'
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
        '--task', default='regression', 
        help='which prediction task to use: [classification, regression]'
    )
    parser.add_argument(
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )

    args = parser.parse_args()
    method = args.method
    train_dataset = args.train_dataset
    eval_dataset = args.eval_dataset
    task = args.task

    args.dataset = eval_dataset #load_data_splits expect this property
    data = load_data_splits(args, splits=[args.split])

    # label_shift function assumes a dataset arg:
    if task == 'classification':
        data = handle_label_shift(args, data)
 
    # Load pretrained model
    ##TODO: define model_path, compute checksum, load model, then eval scores on eval data
    model_path = os.path.join(args.model_path, train_dataset + '_' + method)
    model_path = os.path.join(model_path, 'best_estimator.pkl')
    model, checksum = load_model(model_path) 

    #scores = {
    #    'physionet2019_score': get_physionet2019_scorer(args.label_propagation),
    #    'auroc': SCORERS['roc_auc'],
    #    'average_precision': SCORERS['average_precision'],
    #    'balanced_accuracy': SCORERS['balanced_accuracy'],
    #}
    
    
    # Select split for evaluation:
    split = args.split
    if split == 'validation':
        X_eval = data['X_validation']
        y_eval = data['y_validation']
    elif split == 'test':
        X_eval = data['X_test']
        y_eval = data['y_test']
    else:
        raise ValueError(f'{split} not among the valid eval splits: [validation, test]')

    results = {}
    #cache = {}
    #call = partial(_cached_call, cache)
    #for score_name, scorer in scores.items():
    #    results[score_name] = scorer._score(
    #        call, model, X_eval, y_eval)
    #print(results)
    results['model'] = method
    results['model_path'] = model_path
    results['model_checksum'] = checksum
    results['model_params'] = model.get_params()
    results['dataset_train'] = train_dataset
    results['dataset_eval'] = eval_dataset
    results['split'] = split
    results['rep'] = args.rep 
    results['task'] = task 
 
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
    labels = []
    predictions = []
    scores = []
    times = []
    for pid in ids:
        labels.append(y_eval[pid].values.tolist()) 
        predictions.append(preds.loc[pid][0].tolist())
        scores.append(probas.loc[pid][0].values.tolist()) 
        times.append(y_eval[pid].index.tolist())
    results['labels'] = labels
    results['predictions'] = predictions
    results['scores'] = scores 
    results['times'] = times
    results['label_propagation'] = args.label_propagation #was only applied here for classification 
    
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True) 
    outfile = os.path.join(output_path, f'{method}_{train_dataset}_{eval_dataset}.json')

    #clf obj don't go into json format, remove them:
    for key in ['steps', 'est']:
        results['model_params'].pop(key, None)

    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ in "__main__":
    main()

