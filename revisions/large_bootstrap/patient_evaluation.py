"""Evaluate predictive performance for cached predictions on patient level."""

import argparse
import collections
import functools
import itertools
import json
import os
import pathlib

import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from src.evaluation.sklearn_utils import make_consecutive
from src.evaluation.sklearn_utils import shift_onset_label

from src.evaluation.physionet2019_score import physionet2019_utility


def format_dataset(name):
    """Ensure that we consistently use lower-case dataset names."""
    data_mapping = {
        'MIMICDemo': 'mimic_demo',
        'MIMIC': 'mimic',
        'Hirid': 'hirid',
        'EICU': 'eicu',
        'AUMC': 'aumc',
        'Physionet2019': 'physionet2019',
        'pooled': 'pooled',
        'MIMIC_LOCF': 'mimic',
    }
    if name in data_mapping.keys():
        return data_mapping[name]
    elif ',' in name:
        return name.replace(',','+') 
    elif name in data_mapping.values():
        return name

    raise ValueError(f'{name} not among valid dataset names: {data_mapping}')
    return None


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

    # binarize on patient level
    if pat_level:
        for pat_scores in scores:
            pred = 1 if any([score >= thres for score in pat_scores]) else 0
            result.append(pred)
    # binarize prediction of each timestep
    else:
        result = [
            (np.asarray(pat_scores) >= thres).astype(int)
            for pat_scores in scores
        ]

    return result


def format_check(x, y):
    """Sanity check that two nested lists x,y have identical format."""
    for x_, y_ in zip(x, y):
        assert len(x_) == len(y_)


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
    """Extract first alarm of (patient, time series) lists.

    Assumes x to be a list (patients) of lists (time series). If indices
    are provided, the data in x is extracted for these indices. Else,
    binary predictions are assumed and the indices are identified and
    returned.
    """
    result = []
    if indices:
        # use provided indices
        for pat, index in zip(x, indices):
            if index == -1:
                result.append(np.nan)
            else:
                result.append(pat[index])
        return np.asarray(result)
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
        return np.asarray(result), indices


def get_patient_labels(x):
    """Get patient labels from labels per time step."""
    return np.asarray([int(np.sum(pat) > 0) for pat in x])


def extract_onset_index(x):
    """Return index of sepsis onset or -1 if no onset exists."""
    result = []

    for pat in x:
        assert all([item in [0, 1] for item in pat])
        if np.sum(pat) == 0:
            index = -1
        else:
            index = np.argmax(pat)
        result.append(index)

    return result


def first_alarm_eval(y_true, y_pred, times, thres_percentage):
    """Extract and evaluate prediction and label of first alarm."""
    labels = get_patient_labels(y_true)
    case_mask = labels.astype(bool)
    y_pred, pred_indices = extract_first_alarm(y_pred)
    onset_indices = extract_onset_index(y_true)
    alarm_times = extract_first_alarm(times, indices=pred_indices)
    onset_times = extract_first_alarm(times, indices=onset_indices)
    delta = onset_times[case_mask] - alarm_times[case_mask]

    if np.isnan(delta).sum() == len(delta):
        print(f'No earliness information at {100*thres_percentage} % level of the score thresholds. ')

    # determine proportion of TPs caught earlier than n hours before onset:
    windows = np.arange(11)
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
    #for window in windows:
    #    r[f'proportion_{window}_hours_before_onset'] = (delta > window).sum() / delta.shape[0]
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


def evaluate_threshold(data, labels, shifted_labels, times, thres, thres_percentage, measures):
    """Evaluate threshold-based and patient-based measures.

    Parameters
    ----------
    data
        Dictionary of experiment data.

    labels
        Original labels.

    shifted_labels
        Shifted labels. Notice that the shift can also be `0`, depending
        on the input file that is loaded. For semantic reasons, it makes
        more sense to refer to `shifted_labels` within the code, even in
        these situations.

    times: list of lists
        Increment 'patient hours' for measurements.

    thres: `np.float`
        Threshold between [0,1] for classification, and between
        [min_score, max_score] for regression tasks. Used for a
        per-threshold evaluation of measures.

    measures: dict of callable
        Contains callable evaluation measures to perform the evaluation.
    """
    results = {}
    predictions = apply_threshold(data['scores'], thres, pat_level=False)

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
        # The labels that are used in this function *might* or *might
        # not* be shifted, depending on the data.
        results[name] = func(shifted_labels, predictions)

    for name, func in pat_measures.items():
        output_dict = func(labels, predictions, times, thres_percentage)
        results.update(output_dict)

    return results


def main(
        input_file=None,
        output_file=None, 
        num_steps=100,
        lambda_path='config/lambdas',
        n_jobs=10,
        cost=5,
        from_dict=None,
        drop_percentiles=False,
        used_measures=['pat_eval']
    ):
    """Run evaluation based on user parameters."""
    if from_dict:
        # we are directly passed a dictionary
        d = input_file
    else:
        with open(input_file, 'r') as f:
            d = json.load(f)

    eval_dataset = d['dataset_eval']

    #if isinstance(eval_dataset, list):  # the R jsons had lists of str
    #    eval_dataset = eval_dataset[0]
    eval_dataset = format_dataset(eval_dataset)

    ## handle (and sort) R formatted json:
    #if isinstance(d['ids'][0], str):
    #    # times are sorted wrong:
    #    times = d['times']
    #    ids = [int(i) for i in d['ids']]
    #    perm = np.argsort(ids)
    #    times = np.asarray(times)
    #    # properly sorted times (consistent with scores, labels)
    #    times_p = list(times[perm])
    #    d['times'] = times_p
    #    # also reformat labels to ints:
    #    labels = d['labels']
    #    labels = [[int(x) for x in pat] for pat in labels]
    #    d['labels'] = labels

    # Determine min and max threshold
    score_list = [s for pat in d['scores'] for s in pat]
    if drop_percentiles: # just use min to max for getting thresholds
        # (backwards compatible as we used percentiles before)
        score_max = max(score_list) 
        score_min = min(score_list) 
    else:
        score_max = np.percentile(score_list, 99.5)  # max(score_list)
        score_min = np.percentile(score_list, 0.5)   # min(score_list)

    # all measures
    measures = {
        #'tp_recall': flatten_wrapper(
        #    functools.partial(recall_score, zero_division=0)
        #),
        #'tp_precision': flatten_wrapper(
        #    functools.partial(precision_score, zero_division=0)
        #),
        #'pat_physionet2019_score': utility_score_wrapper(lam=lam),
        'pat_eval': first_alarm_eval
    }
    measures_ = {}
    for measure in used_measures:
        measures_[measure] = measures[measure]
    measures = measures_
    print('Evaluating the following measures: {measures.keys()}')

    n_steps = num_steps
    thresholds = np.linspace(score_min, score_max, n_steps)
    print(f'Using {n_steps} thresholds between {score_min} and {score_max}.')
    results = collections.defaultdict(list)

    # This shared information does not change between different
    # thresholds, so it is sufficient to query it *once*.
    labels = d['labels']
    times = d['times']  # list of lists of incrementing patient hours

    # Check whether label propagation is available in the data or not.
    # If it is available we perform it for all time-based measures.
    shift = 0
    if 'label_propagation' in d.keys():
        shift = d['label_propagation']
        print(f'Using `label_propagation = {shift}` to shift labels')

    shifted_labels = [
        # Slightly convoluted: we want to ensure that the labels are
        # usable in the time-based evaluation, so we need to convert
        # from `pd.Series` to `np.array` again.
        shift_onset_label(
            patient_id,
            pd.Series(y_true, dtype=int, index=time),
            -shift
        ).values
        for patient_id, y_true, time in zip(d['ids'], labels, times)
    ]

    # evaluate thresholds in parallel:
    result_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(evaluate_threshold)(
            d,
            labels,
            shifted_labels,
            times,
            thres,
            i/n_steps,
            measures
        )
        for i, thres in enumerate(thresholds)
    )

    for thres, current in zip(thresholds, result_list):
        results['thres'].append(thres)

        for k, v in current.items():
            if not isinstance(v, np.ndarray):
                results[k].append(v)

    if from_dict:
        return results
    else:
       # Ensures that the directory hierarchy exists for us to write
       # something to the disk.
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
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
        default=100,
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
        '--n_jobs',
        default=10,
        type=int,
        help='number of jobs for parallelizing over thresholds',
    )
    parser.add_argument(
        '--cost',
        default=5,
        type=int,
        help='lambda cost to use (default 0, inactive)'
    )
    parser.add_argument(
        '--drop_percentiles',
        action='store_true',
        help='Flag to not use percentile-based, robust definition of thresholds, but use min to max'
    )
    parser.add_argument(
        '--from-dict',
        action='store_true',
        help='flag to read and return dict without reading and writing files'
    )
    args = parser.parse_args()

    if pathlib.Path(args.output_file).exists() and not args.force:
        raise RuntimeError(f'Refusing to overwrite {args.output_file} unless '
                           f'`--force` is set.')

    main(
        args.input_file,
        args.output_file, 
        args.num_steps,
        args.lambda_path,
        args.n_jobs,
        args.cost,
        args.from_dict,
        args.drop_percentiles,
    )
