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

 
def apply_threshold(scores, thres, pat_level=True):
    """
    applying threshold to scores to return binary predictions
    (assuming list of lists (patients and time series))
    """
    result = []
    for pat_scores in scores:
        if pat_level:
            #binarize on patient level:
            pred = 1 if any([score >= thres for score in pat_scores]) else 0 
            result.append(pred)
        else:
            #binarize prediction of each timestep
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

def extract_first_alarm(x, indices=None):
    """
    assumes x to be a list (patients) of lists (time series)
        - if indices are provided, the data in x is extracted for these indices
            otherwise, a binary predictions are assumed and the indices are 
            identified and returned.
    """
    result = []
    if indices:
        # use provided indices
        for pat, index in zip(x, indices):
            if index == -1:
                result.append(np.nan)
            else:
                result.append(pat[index]) 
        return np.array(result)
    else:
        indices = []
        # extract indices ourselves
        for pat in x:
            # ensure that we have binary predictions 
            assert all([item in [0,1] for item in pat])
                
            # get index of first `1`
            index = np.argmax(pat)
            label = 1 if np.sum(pat) > 0 else 0
            result.append(label)
            if not label: #if no alarm was raised
                index = -1 #distinguish alarm in first hour from no alarm
            indices.append(index)
        return np.array(result), indices


def get_patient_labels(x):
    """ from time step labels"""
    labels = []
    for pat in x:
        label = 1 if np.sum(pat) > 0 else 0
        labels.append(label)
    return np.array(labels)
        
def extract_onset_index(x):
    """ get index of sepsis onset,
        -1 indicates no onset in a patient.
    """
    result = []
    for pat in x:
        assert all([item in [0,1] for item in pat])
        if np.sum(pat) == 0:
            index = -1
        else: 
            index = np.argmax(pat) 
        result.append(index)
    return result 
    
def first_alarm_eval(y_true, y_pred, times):
    """ extract and evaluate prediction and label of first alarm
    """
    labels = get_patient_labels(y_true)
    print(f'Cases: {labels.sum()}')
    print(f'Prevalence: {labels.sum()/len(labels)*100} %')
    case_mask = labels.astype(bool)
    y_pred, pred_indices = extract_first_alarm(y_pred)
    onset_indices = extract_onset_index(y_true)
    alarm_times  = extract_first_alarm(times, indices=pred_indices)
    onset_times  = extract_first_alarm(times, indices=onset_indices)
    r = {} #results 
    r['pat_recall'] = recall_score(labels, y_pred)
    r['pat_precision'] = precision_score(labels, y_pred)
    r['alarm_times'] = alarm_times 
    r['onset_times'] = onset_times
    delta = alarm_times[case_mask] - onset_times[case_mask]
    r['case_delta'] = delta #including nans
    delta_ = delta[~np.isnan(delta)] #excluding nans for statistics
     
    r['control_alarm_times'] = alarm_times[~case_mask]
    r['case_alarm_times'] = alarm_times[case_mask]
    if len(delta_) == 0: 
        print('No non-nan value in delta!')
        r['earliness_mean'] = r['earliness_median'] = r['earliness_min'] = r['earliness_max'] = np.nan
    else:
        r['earliness_mean'] = np.mean(delta_)
        r['earliness_median'] = np.median(delta_)
        r['earliness_min'] = np.min(delta_)
        r['earliness_max'] = np.max(delta_)
    return r 
     
def evaluate_threshold(data, labels, thres, measures):
    """
    function to evaluate eval measures for a given threshold
    - data: dictionary of experiment output data
    - thres: float between [0,1]
    - measures: dict of callable evaluation measures to quantify
    """
    results = {}
    predictions = apply_threshold(data['scores'], thres, pat_level=False)
    times = data['times'] #list of lists of incrementing patient hours

    # sanity check that format fits:
    format_check(predictions, labels)
    format_check(predictions, times)

    # time point measures:
    tp_keys = [key for key in measures.keys() if 'tp_' in key]
    tp_measures = {key: measures[key] for key in tp_keys}
    # patient level measures:
    pat_keys = [key for key in measures.keys() if 'pat_' in key]
    pat_measures = {key: measures[key] for key in pat_keys} 
 
    for name, func in tp_measures.items():
        results[name] = func(labels, predictions) 
    for name, func in pat_measures.items():
        output_dict = func(labels, predictions, times) 
        results.update(output_dict) 
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
    
    measures = {'tp_recall': flatten_wrapper(recall_score), 
                'tp_precision': flatten_wrapper(precision_score),
                'pat_eval': first_alarm_eval
    }
    n_steps = 200
    thresholds = np.arange(0,1,1/n_steps)
    results = {'thres': [], 'pat_recall': [], 'pat_precision': [],
        'earliness_mean': [], 'earliness_median': [], 'tp_precision': [],
        'tp_recall': []}

    for thres in thresholds:
        current = evaluate_threshold(d, labels, thres, measures)
        for key in results.keys():
            if key == 'thres':
                new_val = thres
            else:
                new_val = current[key]
            results[key].append(new_val)
    
    out_file = os.path.split(args.experiment_path)[-1]
    out_file = os.path.join(args.output_path, '_'.join(['patient_eval', out_file])) 
    with open(out_file, 'w') as f:
        json.dump(results, f)

if __name__ in "__main__":
    main()


