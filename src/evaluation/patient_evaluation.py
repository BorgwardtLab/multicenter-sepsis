"""
Script to evaluate predictive performance for cached predictions on a patient-level. 
"""

import argparse
import collections
import itertools
import json

import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix


def fpr(y_true, y_pred):
    """Calculate false positive rate (FPR)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def specificity(y_true, y_pred):
    """Calculate true negative rate (TNR, specificity)."""
    return 1 - fpr(y_true, y_pred)


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


def return_wrapper(fn, index):
    """Return only one value from a function with multiple return values."""
    def wrapped(*args):
        return fn(*args)[index]

    return wrapped


def flatten_list(x):
    """Fully unravel arbitrary list and return it as an `np.array`."""
    return np.asarray(list(itertools.chain(*x)))


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
    

def first_alarm_eval(y_true, y_pred, times, scores):
    """Extract and evaluate prediction and label of first alarm."""
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
    r['pat_specificity'] = specificity(labels, y_pred)
    r['alarm_times'] = alarm_times
    r['onset_times'] = onset_times

    delta = alarm_times[case_mask] - onset_times[case_mask]
    r['case_delta'] = delta  # including nans
    delta_ = delta[~np.isnan(delta)]   # excluding nans for statistics

    r['control_alarm_times'] = alarm_times[~case_mask]
    r['case_alarm_times'] = alarm_times[case_mask]
    if len(delta_) == 0:
        print('No non-nan value in delta!')
        r['earliness_mean']         \
            = r['earliness_median'] \
            = r['earliness_min']    \
            = r['earliness_max']    \
            = np.nan
    else:
        r['earliness_mean'] = np.mean(delta_)
        r['earliness_median'] = np.median(delta_)
        r['earliness_min'] = np.min(delta_)
        r['earliness_max'] = np.max(delta_)
    return r


def evaluate_threshold(data, labels, thres, measures):
    """Evaluate threshold-based and patient-based measures.

    - data: dictionary of experiment output data
    - thres: float between [0,1]
    - measures: dict of callable evaluation measures to quantify
    """
    results = {}
    predictions = apply_threshold(data['scores'], thres, pat_level=False)
    times = data['times']  # list of lists of incrementing patient hours

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
        output_dict = func(labels, predictions, data['scores'], times)
        results.update(output_dict)

    return results


def format_check(x,y):
    """
    sanity check that two nested lists x,y have identical format
    """
    for x_, y_ in zip(x,y):
        assert len(x_) == len(y_)

def main(args):
    
    # TODO: missing capability to deal with unshifted labels; this
    # script will currently just use the labels that are avaialble
    # in the input file.

    #scores = {
    #    'physionet2019_score': get_physionet2019_scorer(args.label_propagation),
    #    'auroc': SCORERS['roc_auc'],
    #    'average_precision': SCORERS['average_precision'],
    #    'balanced_accuracy': SCORERS['balanced_accuracy'],
    #}

    with open(args.input_file, 'r') as f:
        d = json.load(f)

    measures = {
        'tp_recall': flatten_wrapper(recall_score),
        'tp_precision': flatten_wrapper(precision_score),
        'tp_auprc': flatten_wrapper(average_precision_score),
        'pat_eval': first_alarm_eval
    }

    # TODO: make configurable?
    n_steps = 200
    thresholds = np.linspace(0, 1, n_steps)

    results = collections.defaultdict(list)

    for thres in thresholds:
        current = evaluate_threshold(d, d['labels'], thres, measures)

        results['thres'].append(thres)

        for k, v in current.items():
            if not isinstance(v, np.ndarray):
                results[k].append(v)

    # Add measures that are *not* based on any measure and apply to the
    # patient level.

    # FIXME: need to make sure that nothing is overwritten
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ in "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-file',
        required=True,
        type=str,
        help='Path to JSON file for which to run the evaluation',
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Output file path for storing JSON with evaluation results'
    )

    args = parser.parse_args()

    main(args)
