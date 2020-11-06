"""Train sklearn pipeline on dataset."""
import argparse
from functools import partial
import json
import os
from time import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.metrics import SCORERS
from sklearn.metrics._scorer import _cached_call
import joblib

from src.sklearn.data.utils import load_data, load_pickle, save_pickle
from src.evaluation import (
    get_physionet2019_scorer, StratifiedPatientKFold, shift_onset_label)

def load_data_from_input_path(input_path, dataset_name, index):
    """Load the data according to dataset_name, and index-handling

    Returns:
        X_train, X_test, y_train, y_test similar to sklearn train_test_split

    """
    input_path = os.path.join('datasets', dataset_name, input_path)
    data = load_data(path=os.path.join(input_path, 'processed'), index=index)
    return (
        data['X_train'],
        data['X_validation'],
        data['y_train'],
        data['y_validation']
    )


def get_pipeline_and_grid(method_name, clf_params):
    """Get sklearn pipeline and parameter grid."""
    # Convert input format from argparse into a dict
    clf_params = dict(zip(clf_params[::2], clf_params[1::2]))
    if method_name == 'lgbm':
        import lightgbm as lgb
        parameters = {'n_jobs': -1}
        parameters.update(clf_params)
        est = lgb.LGBMClassifier(**parameters)

        pipe = Pipeline(steps=[('est', est)])
        param_dist = {
            'est__n_estimators': [50, 100, 300, 500, 1000],
            'est__boosting_type': ['gbdt', 'dart'],
            'est__learning_rate': [0.001, 0.01, 0.1, 0.5],
            'est__num_leaves': [30, 50, 100],
            'est__scale_pos_weight': [1, 10, 20, 50, 100]
        }
        return pipe, param_dist
    elif method_name == 'xgb':
        from dask_ml.xgboost import XGBClassifier
        parameters = {'n_jobs': 30}
        parameters.update(clf_params)
        est = XGBClassifier(**parameters)

        pipe = Pipeline(steps=[('est', est)])
        
        param_dist = {
            'est__n_estimators': [50, 100, 300, 500, 1000],
            'est__booster': ['gbtree', 'dart'],
            'est__learning_rate': [0.001, 0.01, 0.1, 0.5],
            'est__subsampling': [0.3,0.5,0.9,1],
            'est__scale_pos_weight': [1, 10, 20, 50, 100],
            'est__grow_policy': ['depthwise', 'lossguide']
        }
        return pipe, param_dist        
    elif method_name == 'lr':
        #from dask_ml.linear_model import LogisticRegression as LR
        from sklearn.linear_model import LogisticRegression as LR
        parameters = {'n_jobs': 10}
        parameters.update(clf_params)
        est = LR(**parameters)
        pipe = Pipeline(steps=[('est', est)])
        # hyper-parameter grid:
        param_dist = {
            'est__penalty': ['l2','none'],
            'est__C': np.logspace(-2,2,50),
            #'est__solver': ['gradient_descent','admm'] #when using dask LR 
            'est__solver': ['sag', 'saga'], 
        }
        return pipe, param_dist
    else:
        raise ValueError('Invalid method: {}'.format(method_name))


def apply_label_shift(labels, shift):
    """Apply label shift to labels."""
    labels = labels.copy()
    patients = labels.index.get_level_values('id').unique()
    for patient in patients:
        labels[patient] = shift_onset_label(patient, labels[patient], shift)
    return labels


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
        '--dask', action='store_true', default=False,
        help='use dask backend for grid search parallelism'
    )
    parser.add_argument(
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )

    args = parser.parse_args()

    if args.dask:
        from dask.distributed import Client
        client = Client(n_workers=args.cv_n_jobs, memory_limit='999GB', local_directory='/local0/tmp/dask')
 
    X_train, X_val, y_train, y_val = load_data_from_input_path(
        args.input_path, args.dataset, args.index)
    
    if args.label_propagation != 0:
        # Label shift is normally assumed to be in the direction of the future.
        # For label propagation we should thus take the negative of the
        # provided label propagation parameter
        cached_path = os.path.join('datasets', args.dataset, 'data', 'cached')
        cached_file = os.path.join(cached_path, f'y_shifted_{args.label_propagation}'+'_{}.pkl')
        cached_train = cached_file.format('train')
        cached_validation = cached_file.format('validation')
  
        if os.path.exists(cached_train) and not args.overwrite:
            # read label-shifted data from json:
            print('Loading cached labels shifted by {args.label_propagation} hours')
            y_train = load_pickle(cached_train)
            y_val = load_pickle(cached_validation)
        else:
            # do label-shifting here: 
            start = time()
            y_train = apply_label_shift(y_train, -args.label_propagation)
            y_val = apply_label_shift(y_val, -args.label_propagation)
            elapsed = time() - start
            print(f'Label shift took {elapsed:.2f} seconds')
            #and cache data to quickly reuse from now:
            print('Caching shifted labels..')
            save_pickle(y_train, cached_train) #save pickle also creates folder if needed 
            save_pickle(y_val, cached_validation)

    pipeline, hparam_grid = get_pipeline_and_grid(args.method, args.clf_params)

    scores = {
        'physionet_utility': get_physionet2019_scorer(args.label_propagation),
        'roc_auc': SCORERS['roc_auc'],
        'average_precision': SCORERS['average_precision'],
        'balanced_accuracy': SCORERS['balanced_accuracy'],
    }
    if args.dask:
        from dask_ml.model_selection import RandomizedSearchCV
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=hparam_grid,
            scoring=scores,
            refit='physionet_utility', 
            n_iter=args.n_iter_search,
            cv=StratifiedPatientKFold(n_splits=5),
            iid=False,
            n_jobs=args.cv_n_jobs,
            scheduler=client
        )
        scores = {
            'roc_auc': SCORERS['roc_auc'],
            'average_precision': SCORERS['average_precision'],
            'balanced_accuracy': SCORERS['balanced_accuracy'],
        }
    else:
        from sklearn.model_selection import RandomizedSearchCV
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
    start = time()
    if args.dask:
        #dask dataframe:
        import dask.dataframe as dd
        X_train = dd.from_pandas(X_train, npartitions=1000, sort=True)
        y_train = dd.from_pandas(y_train, npartitions=1000, sort=True)
        random_search.fit(X_train, y_train)
        # TODO:try out dask df here too 
    else:
        random_search.fit(X_train, y_train)
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
            call, best_estimator, X_val, y_val)
    print(results)
    results['method'] = args.method
    results['best_params'] = random_search.best_params_
    results['n_iter_search'] = args.n_iter_search
    results['runtime'] = elapsed
    for method in ['predict', 'predict_proba', 'decision_function']:
        try:
            results['val_' + method] = call(
                best_estimator, method, X_val).tolist()
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
