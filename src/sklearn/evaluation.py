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
from src.sklearn.main import load_data_from_input_path, apply_label_shift


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
        '--train_dataset', default='physionet2019',
        help='Train Dataset Name: [physionet2019, ..]'
    )
    parser.add_argument(
        '--eval_dataset', default='physionet2019',
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
        '--clf_params', nargs='+', default=[],
        help='Parameters passed to the classifier constructor'
    )
    parser.add_argument(
        '--feature_set', default='all',
        help='which feature set should be used: [all, challenge], where challenge refers to the subset as derived from physionet challenge variables'
    )
    parser.add_argument(
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )

    args = parser.parse_args()

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
            print(f'Loading cached labels shifted by {args.label_propagation} hours')
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

    # Load pretrained model
    ##TODO: define model_path, compute checksum, load model, then eval scores on eval data
    checksum = hashlib.md5(open(model_path + 'best_estimator.pkl','rb').read()).hexdigest() 
    
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

