"""Run feature selection for a given sklearn model."""
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
import matplotlib.pyplot as plt
#from sklearn.feature_selection import RFECV

#custom modules
from src.sklearn.data.utils import load_data, load_pickle, save_pickle
from src.evaluation import (
    get_physionet2019_scorer, StratifiedPatientKFold, shift_onset_label)
from src.sklearn.main import load_data_from_input_path, handle_label_shift 
from ._rfe import RFECV

def get_pipeline(model_path):
    """Get sklearn pipeline and parameter grid."""
    with open(model_path, 'rb') as f:
        pipe = joblib.load(f)
    print('Loading the following pipeline:', pipe)
    return pipe

def main():
    """Parse arguments and launch fitting of model."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_path', default='data/sklearn',
        help='Path to input data directory (relative from dataset directory)'
    )
    parser.add_argument(
        '--result_path', default='results/feature_selection',
        help='Relative path to where FS results are written to'
    )
    parser.add_argument(
        '--dataset', default='aumc',
        help='Dataset Name: [aumc, physionet2019, ..]'
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
        help='Method which is used (only for file naming)'
    )
    parser.add_argument(
        '--model_path', 
        default="""results/feature_selection/signatures_wavelets_counts_GAIN/aumc_lgbm/best_estimator.pkl""", 
        type=str,
        help='Path to pretrained best estimator'
    )
    parser.add_argument(
        '--cv_n_jobs', type=int, default=10,
        help='n_jobs for cross-validation'
    )
    parser.add_argument(
        '--feature_set', default='all',
        help="""which feature set should be used: [all, challenge], CAVE: this needs to be 
                consistent with the saved model in model_path"""
    )
    parser.add_argument(
        '--extended_features', default=True, type=bool,
        help='flag if extended feature set should be used (incl measurement counter, wavelets etc)'
    )
    parser.add_argument(
        '--step', type=float, default=1,
        help='how many features are eliminated per round'
    )

    parser.add_argument(
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )

    args = parser.parse_args()

    data = load_data_from_input_path(
        args.input_path, args.dataset, args.index, args.extended_features)

    data = handle_label_shift(args, data)
 
    pipeline = get_pipeline(args.model_path)

    scores = {
        'physionet_utility': get_physionet2019_scorer(args.label_propagation),
        'roc_auc': SCORERS['roc_auc'],
        'average_precision': SCORERS['average_precision'],
        'balanced_accuracy': SCORERS['balanced_accuracy'],
    }
  
    # recursive feature elimination cross-validation 
    rfe = RFECV(
        pipeline['est'],
        scoring = scores['physionet_utility'],
        cv = StratifiedPatientKFold(n_splits=3), #FIXME: RFECV converts data to np.array! doesn't work without index..
        n_jobs = args.cv_n_jobs,
        step = args.step
    ) 
    
    # actually run the feature selection 
    start = time()
    #from IPython import embed; embed()
    rfe.fit(data['X_train'], data['y_train'])
    elapsed = time() - start
    print(
        "Feature selection took %.2f seconds"
        " parameter settings." % (elapsed)
    )
    result_path = os.path.join(args.result_path, 'feature_selection_' + args.dataset+'_'+args.method)
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, 'rfe.pkl'), 'wb') as f:
        joblib.dump(rfe, f, compress=1) 
    
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
    plt.savefig(os.path.join(result_path, 'grid_scores.png'), dpi=300)
    
    # TODO: check selected # of features, check performance etc
 
    #cv_results.to_csv(os.path.join(result_path, 'cv_results.csv'))

    ## Quantify performance on validation split
    #best_estimator = random_search.best_estimator_
    #results = {}
    #cache = {}
    #call = partial(_cached_call, cache)
    #for score_name, scorer in scores.items():
    #    results['val_' + score_name] = scorer._score(
    #        call, best_estimator, data['X_validation'], data['y_validation'])
    #print(results)
    #results['method'] = args.method
    #results['best_params'] = random_search.best_params_
    #results['n_iter_search'] = args.n_iter_search
    #results['runtime'] = elapsed
    #for method in ['predict', 'predict_proba', 'decision_function']:
    #    try:
    #        results['val_' + method] = call(
    #            best_estimator, method, data['X_validation']).tolist()
    #    except AttributeError:
    #        # Not all estimators support all methods
    #        continue

    #with open(os.path.join(result_path, 'results.json'), 'w') as f:
    #    json.dump(results, f)
    #joblib.dump(
    #    best_estimator,
    #    os.path.join(result_path, 'best_estimator.pkl'),
    #    compress=1
    #)


if __name__ in "__main__":
    main()
