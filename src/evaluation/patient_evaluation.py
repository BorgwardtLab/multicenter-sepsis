"""Evaluate predictive performance for cached predictions on patient level."""

import argparse
import collections
import functools
import itertools
import json
import os
import pandas as pd
import pathlib
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from src.evaluation.sklearn_utils import make_consecutive
from src.evaluation.physionet2019_score import physionet2019_utility



def fpr(y_true, y_pred):
    """Calculate false positive rate (FPR)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def specificity(y_true, y_pred):
    """Calculate true negative rate (TNR, specificity)."""
    return 1 - fpr(y_true, y_pred)


def apply_threshold(scores, thres, pat_level=True):
    """Threshold scores to return binary predictions.

    This function applies a given threshold to scores to return binary
    predictions, assuming list of lists (patients and time series) as an
    input.
    """
    result = []
    for pat_scores in scores:
        if pat_level:
            # binarize on patient level:
            pred = 1 if any([score >= thres for score in pat_scores]) else 0
            result.append(pred)
        else:
            # binarize prediction of each timestep
            preds = (pat_scores >= thres).astype(int)
            result.append(preds)

    return result


def flatten_list(x):
    """Fully unravel arbitrary list and return it as an `np.array`."""
    return np.asarray(list(itertools.chain(*x)))


def flatten_wrapper(func):
    """Apply given function to flattened lists."""
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
            assert all([item in [0, 1] for item in pat])

            # get index of first `1`
            index = np.argmax(pat)
            label = 1 if np.sum(pat) > 0 else 0
            result.append(label)
            if not label:  # if no alarm was raised
                index = -1   # distinguish alarm in first hour from no alarm
            indices.append(index)
        return np.array(result), indices


def get_patient_labels(x):
    """Get patient labels from labels per time step."""
    return np.asarray([int(np.sum(pat) > 0) for pat in x])


def extract_onset_index(x):
    """ get index of sepsis onset,
        -1 indicates no onset in a patient.
    """
    result = []

    for pat in x:
        assert all([item in [0, 1] for item in pat])
        if np.sum(pat) == 0:
            index = -1
        else:
            index = np.argmax(pat)
        result.append(index)

    return result


def first_alarm_eval(y_true, y_pred, times):
    """Extract and evaluate prediction and label of first alarm."""
    labels = get_patient_labels(y_true)
    print(f'Cases: {labels.sum()}')
    print(f'Prevalence: {labels.sum()/len(labels)*100:.2f}%')
    case_mask = labels.astype(bool)
    y_pred, pred_indices = extract_first_alarm(y_pred)
    onset_indices = extract_onset_index(y_true)
    alarm_times = extract_first_alarm(times, indices=pred_indices)
    onset_times = extract_first_alarm(times, indices=onset_indices)
    delta = alarm_times[case_mask] - onset_times[case_mask]

    r = {
        'pat_recall': recall_score(labels, y_pred, zero_division=0),
        'pat_precision': precision_score(labels, y_pred, zero_division=0),
        'pat_specificity': specificity(labels, y_pred),
        'alarm_times': alarm_times,
        'onset_times': onset_times,
        'case_delta': delta,
        'control_alarm_times': alarm_times[~case_mask],
        'case_alarm_times': alarm_times[case_mask],
        'earliness_mean': np.nanmean(delta),
        'earliness_median': np.nanmedian(delta),
        'earliness_min': np.nanmin(delta),
        'earliness_max': np.nanmax(delta),
    }

    return r

def utility_score_wrapper(lam=1, **kwargs):
    """ returns a wrapped util score function which can 
        process list of lists of labels, predictions and times.
    """
    def wrapped_func(y_true, y_pred, times):
        labels = []; preds = []
        for i in np.arange(len(y_true)):
            l = y_true[i]; p = y_pred[i]; t = times[i]
            df = pd.DataFrame( [ l, p, t ] ).T
            df.columns = ['label','pred','time']
            df = df.set_index('time')
            n_nans = df.isnull().sum().sum()
            assert n_nans == 0
            # make time series is consecutive:
            label, pred = make_consecutive(
                df['label'], df['pred'])
            labels.append(label.values)
            preds.append(pred.values)
        score = physionet2019_utility(labels, preds, lam, **kwargs)
        return {'physionet2019_utility': score} 
    return wrapped_func 

def evaluate_threshold(data, thres, measures):
    """Evaluate threshold-based and patient-based measures.

    - data: dictionary of experiment output data
    - thres: float between [0,1] for classification, and between [min_score, max_score] 
        for regression
    - measures: dict of callable evaluation measures to quantify
    """
    results = {}
    predictions = apply_threshold(data['scores'], thres, pat_level=False)
    labels = data['labels']
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
        output_dict = func(labels, predictions, times)
        results.update(output_dict)

    return results


def format_check(x, y):
    """Sanity check that two nested lists x,y have identical format."""
    for x_, y_ in zip(x, y):
        assert len(x_) == len(y_)


def main(args):
    """Run evaluation based on user parameters."""
    # TODO: missing capability to deal with unshifted labels; this
    # script will currently just use the labels that are available
    # in the input file.
   
    # Load input json 
    with open(args.input_file, 'r') as f:
        d = json.load(f)

    # Determine lambda path:
    lambda_path = os.path.join(args.lambda_path, 
        'lambda_{}_rep_{}.json' ) 
    # Compute aggregate lambda over all train reps:
    lambdas = []
    eval_dataset = d['dataset_eval']
    for rep in np.arange(5):
        with open(lambda_path.format(eval_dataset, rep), 'r') as f:
            lam = json.load(f)['lam']
            lambdas.append(lam)
    lam = np.mean(lambdas)
    print(f'Using aggregated lambda: {lam} from the {eval_dataset} dataset')  
   
    # Determine min and max threshold 
    if args.task == 'regression':
        score_list = [s for pat in d['scores'] for s in pat]
        score_max = np.percentile(score_list, 99.5) #max(score_list)
        score_min = np.percentile(score_list, 0.5) #min(score_list)
    else:
        score_max = 1; score_min = 0

    measures = {
        'tp_recall': flatten_wrapper(
            functools.partial(recall_score, zero_division=0)
        ),
        'tp_precision': flatten_wrapper(
            functools.partial(precision_score, zero_division=0)
        ),
        #TODO: if we only have shifted labels we may need to pass the argument here
        'pat_physionet2019_score' : utility_score_wrapper(lam=lam),
        'pat_eval': first_alarm_eval
    }

    n_steps = args.num_steps
    thresholds = np.linspace(score_min, score_max, n_steps)
    print(f'Using {n_steps} thresholds between {score_min} and {score_max}.')
    results = collections.defaultdict(list)

    # evaluate thresholds in parallel:
    result_list = Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(evaluate_threshold)(d, thres, measures) for thres in thresholds) 
 
    for thres, current in zip(thresholds, result_list):
        #current = evaluate_threshold(d, thres, measures)
        results['thres'].append(thres)

        for k, v in current.items():
            if not isinstance(v, np.ndarray):
                results[k].append(v)

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
    parser.add_argument(
        '-s', '--num-steps',
        default=200,
        type=int,
        help='Number of steps for thresholding'
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='If set, overwrites output files'
    )
    parser.add_argument(
        '--lambda_path', 
        help='path to lambda file', 
        default='config/lambdas'
    )
    parser.add_argument(
        '--task', 
        help='prediction task [regression, classification]', 
        default='regression'
    )
    parser.add_argument(
        '--n_jobs', 
        help='number of jobs for parallelizing over thresholds', 
        default=10, type=int
    )


    args = parser.parse_args()

    if pathlib.Path(args.output_file).exists() and not args.force:
        raise RuntimeError(f'Refusing to overwrite {args.output_file} unless '
                           f'`--force` is set.')

    main(args)
