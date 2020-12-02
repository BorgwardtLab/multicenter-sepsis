"""Train sklearn pipeline on dataset."""
import argparse
from functools import partial
import json
import os
from time import time
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import SCORERS
from sklearn.metrics._scorer import _cached_call
from sklearn.model_selection import RandomizedSearchCV
#custom modules
from src.sklearn.data.utils import load_data, load_pickle, save_pickle
from src.evaluation import (
    get_physionet2019_scorer, StratifiedPatientKFold, shift_onset_label)

def load_data_from_input_path(input_path, dataset_name, index, extended_features=False):
    """Load the data according to dataset_name, and index-handling

    Returns dict with keys:
        X_train, X_validation, X_test, y_train, y_validation, y_test 

    """
    input_path = os.path.join('datasets', dataset_name, input_path)
    data = load_data(   path=os.path.join(input_path, 'processed'),
                        index=index, extended_features=extended_features)
    return data 

def get_pipeline_and_grid(method_name, clf_params, feature_set):
    """Get sklearn pipeline and parameter grid."""
    # first determine which feature set to use for current model:
    steps = [] #pipeline steps
    if feature_set == 'challenge':
        from src.sklearn.data.subsetters import ChallengeFeatureSubsetter
        subsetter = ChallengeFeatureSubsetter() #transform to remove all features which cannot be derived from challenge data
        steps.append(('feature_subsetter', subsetter)) 
    elif feature_set != 'all':
        raise ValueError(f'provided feature set {feature_set} is not among the valid [all, challenge]')
 
    # Convert input format from argparse into a dict
    clf_params = dict(zip(clf_params[::2], clf_params[1::2]))
    if method_name == 'lgbm':
        import lightgbm as lgb
        parameters = {'n_jobs': -1}
        parameters.update(clf_params)
        est = lgb.LGBMClassifier(**parameters)
        steps.append(('est', est))
        pipe = Pipeline(steps)
        param_dist = {
            'est__n_estimators': [50, 100, 300, 500, 1000],
            'est__boosting_type': ['gbdt', 'dart'],
            'est__learning_rate': [0.001, 0.01, 0.1, 0.5],
            'est__num_leaves': [30, 50, 100],
            'est__scale_pos_weight': [1, 10, 20, 50, 100]
        }
        return pipe, param_dist
    elif method_name == 'lr':
        from sklearn.linear_model import LogisticRegression as LR
        parameters = {'n_jobs': 10}
        parameters.update(clf_params)
        est = LR(**parameters)
        steps.append(('est', est))
        pipe = Pipeline(steps)
        # hyper-parameter grid:
        param_dist = {
            'est__penalty': ['l2','none'],
            'est__C': np.logspace(-2,2,50),
            'est__solver': ['sag', 'saga'], 
        }
        return pipe, param_dist
    else:
        raise ValueError('Invalid method: {}'.format(method_name))

def apply_label_shift(labels, shift):
    """Apply label shift to labels."""
    labels = labels.copy()
    patients = labels.index.get_level_values('id').unique()
    # sanity check: assert that no reordering occured:
    assert np.all(labels.index.levels[0] == patients)
    new_labels = pd.DataFrame() 
    
    for patient in patients:
        #labels[patient] = shift_onset_label(patient, labels[patient], shift)
        # the above (nice) solution lead to pandas bug in newer version.. 
        shifted_labels = shift_onset_label(patient, labels[patient], shift)
        df = shifted_labels.to_frame()
        df = df.rename(columns={0: 'sep3'}) 
        df['id'] = patient
        new_labels = new_labels.append(df)
    new_labels.reset_index(inplace=True)
    new_labels.set_index(['id', 'time'], inplace=True)
    return new_labels['sep3']

def handle_label_shift(args, d):
    """Handle label shift given argparse args and data dict d"""
    if args.label_propagation != 0:
        # Label shift is normally assumed to be in the direction of the future.
        # For label propagation we should thus take the negative of the
        # provided label propagation parameter
        cached_path = os.path.join('datasets', args.dataset, 'data', 'cached')
        cached_file = os.path.join(cached_path, f'y_shifted_{args.label_propagation}'+'_{}.pkl')
        cached_train = cached_file.format('train')
        cached_validation = cached_file.format('validation')
        cached_test = cached_file.format('test')
 
        if os.path.exists(cached_train) and not args.overwrite:
            # read label-shifted data from json:
            print(f'Loading cached labels shifted by {args.label_propagation} hours')
            y_train = load_pickle(cached_train)
            y_val = load_pickle(cached_validation)
            y_test = load_pickle(cached_test)
        else:
            # unpack dict
            y_train = d['y_train']
            y_val = d['y_validation'] 
            y_test = d['y_test']

            # do label-shifting here: 
            start = time()
            y_train = apply_label_shift(y_train, -args.label_propagation)
            y_val = apply_label_shift(y_val, -args.label_propagation)
            y_test = apply_label_shift(y_test, -args.label_propagation)

            elapsed = time() - start
            print(f'Label shift took {elapsed:.2f} seconds')
            #and cache data to quickly reuse from now:
            print('Caching shifted labels..')
            save_pickle(y_train, cached_train) #save pickle also creates folder if needed 
            save_pickle(y_val, cached_validation)
            save_pickle(y_test, cached_test)
        # update the shifted labels in data dict:
        d['y_train'] = y_train
        d['y_validation'] = y_val
        d['y_test'] = y_test 
    else:
        print('No label shift applied.')
    return d

def main():
    """Parse arguments and launch fitting of model."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_path', default='data/sklearn',
        help='Path to input data directory (relative from dataset directory)'
    )
    parser.add_argument(
        '--result_path', default='results',
        help='Relative path to experimental results (from input path)'
    )
    parser.add_argument(
        '--dataset', default='physionet2019',
        help='Dataset Name: [physionet2019, ..]'
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
        '--clf_params', nargs='+', default=[],
        help='Parameters passed to the classifier constructor'
    )
    parser.add_argument(
        '--n_iter_search', type=int, default=20,
        help='Number of iterations in randomized hyperparameter search')
    parser.add_argument(
        '--cv_n_jobs', type=int, default=10,
        help='n_jobs for cross-validation'
    )
    parser.add_argument(
        '--feature_set', default='all',
        help='which feature set should be used: [all, challenge], where challenge refers to the subset as derived from physionet challenge variables'
    )
    parser.add_argument(
        '--extended_features', default=False, action='store_true',
        help='flag if extended feature set should be used (incl measurement counter, wavelets etc)'
    )
    parser.add_argument(
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )

    args = parser.parse_args()

    data = load_data_from_input_path(
        args.input_path, args.dataset, args.index, args.extended_features)

    data = handle_label_shift(args, data)
 
    pipeline, hparam_grid = get_pipeline_and_grid(args.method, args.clf_params, args.feature_set)

    scores = {
        'physionet_utility': get_physionet2019_scorer(args.label_propagation),
        'roc_auc': SCORERS['roc_auc'],
        'average_precision': SCORERS['average_precision'],
        'balanced_accuracy': SCORERS['balanced_accuracy'],
    }
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=hparam_grid,
        scoring=scores,
        refit='physionet_utility', #'average_precision'
        n_iter=args.n_iter_search,
        cv=StratifiedPatientKFold(n_splits=5),
        iid=False,
        n_jobs=args.cv_n_jobs
    )
    # actually run the randomized search
    start = time()
    random_search.fit(data['X_train'], data['y_train'])
    elapsed = time() - start
    print(
        "RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((elapsed), args.n_iter_search)
    )
    result_path = os.path.join(args.result_path, args.dataset+'_'+args.method)
    os.makedirs(result_path, exist_ok=True)

    cv_results = pd.DataFrame(random_search.cv_results_)
    cv_results.to_csv(os.path.join(result_path, 'cv_results.csv'))

    # Quantify performance on validation split
    best_estimator = random_search.best_estimator_
    results = {}
    cache = {}
    call = partial(_cached_call, cache)
    for score_name, scorer in scores.items():
        results['val_' + score_name] = scorer._score(
            call, best_estimator, data['X_validation'], data['y_validation'])
    print(results)
    results['method'] = args.method
    results['best_params'] = random_search.best_params_
    results['n_iter_search'] = args.n_iter_search
    results['runtime'] = elapsed
    for method in ['predict', 'predict_proba', 'decision_function']:
        try:
            results['val_' + method] = call(
                best_estimator, method, data['X_validation']).tolist()
        except AttributeError:
            # Not all estimators support all methods
            continue

    with open(os.path.join(result_path, 'results.json'), 'w') as f:
        json.dump(results, f)
    joblib.dump(
        best_estimator,
        os.path.join(result_path, 'best_estimator.pkl'),
        compress=1
    )


if __name__ in "__main__":
    main()
