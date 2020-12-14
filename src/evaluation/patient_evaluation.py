"""
Script to evaluate predictive performance for cached predictions on a patient-level. 
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
from sklearn.metrics import recall_score, precision_score

from sklearn.model_selection import RandomizedSearchCV
#custom:
from src.sklearn.main import load_data_from_input_path

 
def apply_threshold(scores, thres):
    """
    applying threshold to scores to return binary predictions
    (assuming list of lists (patients and time series))
    """
    result = []
    for pat_scores in scores:
        preds = []
        for score in pat_scores:
            preds.append(1 if score >= thres else 0)
        result.append(preds)
    return result 

def flatten_list(x):
    """
    some evals require a flattened array, not a nested list
    """
    f = []
    for pat in x:
        for step in pat:
            f.append(step)
    return np.array(f)

def flatten_wrapper(func):
    """ due to nested list format, slightly customize sklearn metrics
        funcs:
    """
    def wrapped(y_true, y_pred):    
        y_true = flatten_list(y_true)
        y_pred = flatten_list(y_pred)
        assert y_true.shape == y_pred.shape
        return func(y_true, y_pred)
    return wrapped
   
def evaluate_threshold(data, labels, thres, measures):
    """
    function to evaluate eval measures for a given threshold
    - data: dictionary of experiment output data
    - thres: float between [0,1]
    - measures: dict of callable evaluation measures to quantify
    """
    results = {}
    predictions = apply_threshold(data['scores'], thres)

    #sanity check that format fits:
    format_check(predictions, labels)

    for name, func in measures.items():
        results[name] = func(labels, predictions)  
    return results
 
def format_check(x,y):
    """
    sanity check that two nested lists x,y have identical format
    """
    for x_, y_ in zip(x,y):
        assert len(x_) == len(y_)

def main():
    """Parse arguments and launch fitting of model."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_path', default='data/sklearn',
        help='Path to input data (relative from dataset directory)'
    )
    parser.add_argument(
        '--experiment_path', default='results/evaluation',
        help='Relative path to experimental results including trained models'
    )
    parser.add_argument(
        '--output_path', default='results/evaluation',
        help='Relative path to evaluation results'
    )
    parser.add_argument(
        '--eval_dataset', default='demo',
        help='Evaluation Dataset Name: [physionet2019, ..]'
    )
    #parser.add_argument(
    #    '--label-propagation', default=6, type=int,
    #    help='By how many hours to shift label into the past. Default: 6'
    #)
    parser.add_argument(
        '--split', default='validation', 
        help='on which split to evaluate [validation (default), test]'
    )
    parser.add_argument(
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )


    args = parser.parse_args()

    data = load_data_from_input_path(
        args.input_path, args.eval_dataset, args.index)
    
    # Recover orginal, unshifted labels: 
    split = args.split
    if split == 'validation':
        y_eval = data['y_validation']
    elif split == 'test':
        y_eval = data['y_test']
    else:
        raise ValueError(f'{split} not among the valid eval splits: [validation, test]')
    ids = y_eval.index.get_level_values('id').unique().tolist()
    labels = []
    for pid in ids:
        labels.append(y_eval[pid].values.tolist())   
     
    #scores = {
    #    'physionet2019_score': get_physionet2019_scorer(args.label_propagation),
    #    'auroc': SCORERS['roc_auc'],
    #    'average_precision': SCORERS['average_precision'],
    #    'balanced_accuracy': SCORERS['balanced_accuracy'],
    #}
   
    # load cached experiment output data into dict d: 
    with open(args.experiment_path, 'r') as f:
        d = json.load(f)
    
    #compare shifted vs unshifted labels:
    s1 = np.sum(np.sum(d['labels']))
    s2 = np.sum(np.sum(labels)) 
    print(f'sum shifted / unshifted labels: {s1} /  {s2} ')
    
    measures = {'recall': flatten_wrapper(recall_score), 
                'precision': flatten_wrapper(precision_score)
    }
    thres = 0.3
    results = evaluate_threshold(d, labels, thres, measures) 
    
    from IPython import embed; embed()

if __name__ in "__main__":
    main()


