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
from sklearn.pipeline import Pipeline
from sklearn.metrics import SCORERS
from sklearn.metrics._scorer import _cached_call
from sklearn.model_selection import RandomizedSearchCV
#custom modules
from src.sklearn.data.utils import load_data, load_pickle, save_pickle
from src.evaluation import (
    get_physionet2019_scorer, StratifiedPatientKFold, shift_onset_label)
from src.sklearn.main import load_data_from_input_path, handle_label_shift

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
        '--input_path', default='data/sklearn',
        help='Path to input data (relative from dataset directory)'
    )
    parser.add_argument(
        '--model_path', default='results/hypersearch4',
        help='Relative path to experimental results including trained models'
    )
    parser.add_argument(
        '--output_path', default='results/evaluation',
        help='Relative path to evaluation results'
    )
    parser.add_argument(
        '--train_dataset', default='physionet2019',
        help='Train Dataset Name: [physionet2019, ..]'
    )
    parser.add_argument(
        '--eval_dataset', default='mimic3',
        help='Evaluation Dataset Name: [physionet2019, ..]'
    )
    parser.add_argument(
        '--label-propagation', default=6, type=int,
        help='By how many hours to shift label into the past. Default: 6'
    )
    parser.add_argument(
        '--overwrite', action='store_true', default=False,
        help='<Currently inactive> To overwrite existing cached data'
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
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )

    args = parser.parse_args()
    method = args.method
    train_dataset = args.train_dataset
    eval_dataset = args.eval_dataset

    data = load_data_from_input_path(
        args.input_path, eval_dataset, args.index)
  
    # label_shift function assumes a dataset arg:
    args.dataset = eval_dataset 
    data = handle_label_shift(args, data)
 
    # Load pretrained model
    ##TODO: define model_path, compute checksum, load model, then eval scores on eval data
    model_path = os.path.join(args.model_path, train_dataset + '_' + method)
    model_path = os.path.join(model_path, 'best_estimator.pkl')
    model, checksum = load_model(model_path) 

    scores = {
        'physionet2019_score': get_physionet2019_scorer(args.label_propagation),
        'auroc': SCORERS['roc_auc'],
        'average_precision': SCORERS['average_precision'],
        'balanced_accuracy': SCORERS['balanced_accuracy'],
    }
    
    
    
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

    # CAME UNTIL HERE  
    results = {}
    cache = {}
    call = partial(_cached_call, cache)
    for score_name, scorer in scores.items():
        results[score_name] = scorer._score(
            call, model, X_eval, y_eval)
    print(results)
    results['model'] = method
    results['model_path'] = model_path
    results['model_checksum'] = checksum
    results['model_params'] = model.get_params()
    results['dataset_train'] = train_dataset
    results['dataset_eval'] = eval_dataset
    results['split'] = split
    
    results['predictions'] = model.predict(X_eval).tolist() 
    results['scores'] = model.predict_proba(X_eval)[:,1].tolist()
    ids = y_eval.index.get_level_values('id').unique().tolist() 
    results['ids'] = ids
    labels = []
    for pid in ids:
        labels.append(y_eval[pid].values.tolist()) 
    results['labels'] = labels
 
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True) 
    outfile = os.path.join(output_path, f'{method}_{train_dataset}_{eval_dataset}.json')

    #clf obj don't go into json format, remove them:
    for key in ['steps', 'est']:
        results['model_params'].pop(key, None)

    from IPython import embed; embed() 
    with open(outfile, 'w') as f:
        json.dump(results, f)
     
if __name__ in "__main__":
    main()

