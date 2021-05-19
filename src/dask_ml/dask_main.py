"""Train sklearn pipeline on dataset."""
import argparse
from functools import partial
import json
import os
import pathlib
from time import time
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import SCORERS
from sklearn.metrics._scorer import _cached_call
from dask_ml.model_selection import RandomizedSearchCV

from src.variables.mapping import VariableMapping
from src.sklearn.data.utils import load_pickle, save_pickle
from src.sklearn.loading import load_and_transform_data
from src.sklearn.shift_utils import handle_label_shift
from src.evaluation import (
    get_physionet2019_scorer, StratifiedPatientKFold)

VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)

def load_data(args, split):
    """
    util function to load current data split
    """
    dataset = args.dataset
    rep = args.rep
    input_path = args.input_path.format(dataset)
    normalizer_path = os.path.join(args.normalizer_path, 
        f'normalizer_{dataset}_rep_{rep}.json' )
    if args.cost > 0:
        lam_file = f'lambda_{dataset}_rep_{rep}_cost_{args.cost}.json'
    else:
        lam_file = f'lambda_{dataset}_rep_{rep}.json'
    lambda_path = os.path.join(args.lambda_path, 
        lam_file )
    split_path = os.path.join(args.split_path, 
        f'splits_{dataset}.json' ) 

    # Load data and apply on-the-fly transforms:
    return load_and_transform_data(
        input_path,
        split_path,
        normalizer_path,
        lambda_path,
        args.feature_path,
        split=split,
        rep=rep,
        feature_set=args.feature_set,
        variable_set=args.variable_set,
        task=args.task,
        baselines=False
    )

def load_data_splits(args, 
    splits=['train','validation','test']):
    """
    Util function to read all 3 data splits of current repetion
    and return dictionary {X_train: [], y_train: [], ..}
    setting the label to the current task
    Parameters: 
    args: config object with properties:
        .task regression or classification
        .index: multi oder single index
        .input_path: path to input data, with dataset name generic with {}
        .dataset: name of dataset
        .split_path: path to split info
        .lambda_path: path to lambdas
        .normalizer_path: path to normalizer
        .features_path: path to feature columns file
        .variable_set: full, physionet
        .feature_set: large, small
        .rep: repetition/fold of split
    """ 
    d = {}
    if args.task == 'classification':
        label = VM_DEFAULT('label')
    elif args.task == 'regression':
        label = VM_DEFAULT('utility')
    else:
        raise ValueError(f'Task {args.task} not among valid tasks. ')
    for split in splits:
        data, lam = load_data(args, split)
        if args.index == 'multi':   
            data = data.reset_index().set_index(
                [VM_DEFAULT('id'), VM_DEFAULT('time')]
            ) 
        d[f'y_{split}'] = data[label]
        # shifted and unshifted labels for down stream eval irrespecive of task:
        d[f'tp_labels_{split}'] = data[VM_DEFAULT('label')]
        d[f'tp_labels_shifted_{split}'] = data[VM_DEFAULT('label')]
        data = data.drop(columns=[ 
            VM_DEFAULT('label'), VM_DEFAULT('utility')
            ], errors='ignore'
        )
        #if args.task == 'regression':
        #    data = data.drop(columns=[label])
        # sanity check as we must not leak any label info to the input data
        assert all( 
            [ VM_DEFAULT(x) not in data.columns 
                for x in ['label', 'utility'] 
            ]
        )
        d[f'X_{split}'] = data 
    return d, lam

def get_pipeline_and_grid(args):
    """Get sklearn pipeline and parameter grid."""
    # first determine which feature set to use for current model:
    # unpack arguments:
    method_name = args.method
    clf_params = args.clf_params 
    task = args.task
 
    # Convert input format from argparse into a dict
    clf_params = dict(zip(clf_params[::2], clf_params[1::2]))
    if method_name == 'lgbm':
        import lightgbm as lgb
        parameters = {'n_jobs': 30}
        parameters.update(clf_params)
        if task == 'classification':
            est = lgb.LGBMClassifier(**parameters)
        elif task == 'regression':
            est = lgb.LGBMRegressor(**parameters)
        else:
            raise ValueError(f'task {task} must be classification or regression') 
        steps.append(('est', est))
        pipe = Pipeline(steps)
        param_dist = {
            'est__n_estimators': [100, 300, 500, 1000, 2000],
            'est__boosting_type': ['gbdt', 'dart'],
            'est__learning_rate': [0.001, 0.01, 0.1, 0.5],
            'est__num_leaves': [30, 50, 100],
            'est__reg_alpha': [0,0.1,0.5,1,3,5], 
        }
        #if args.task == 'classification':
        #    param_dist['est__scale_pos_weight'] = [1, 10, 20, 50, 100]
        return pipe, param_dist

    # TODO: add baselines (and try LogReg) also
    elif method_name == 'lr': #logistic/linear regression
        
        if task == 'classification':
            from src.dask_ml.glm import LogisticRegression as LogReg
            parameters = {'n_jobs': 1, 'solver_kwargs':{'normalize':False} } #10 for non-eicu #-1 led to OOM
            parameters.update(clf_params)
            est = LogReg(**parameters)
            # hyper-parameter grid:
            param_dist = {
                'penalty': ['l1'],
                'C': np.logspace(-3,2,50),
                'solver': ['admm'], #‘admm’, ‘lbfgs’ and ‘proximal_grad’  
            }
        #elif task == 'regression':
        #    from sklearn.linear_model import Lasso 
        #    parameters = {}
        #    parameters.update(clf_params)
        #    est = Lasso(**parameters)
        #    # hyper-parameter grid:
        #    param_dist = {
        #        'est__alpha': np.logspace(-3,2,50),
        #    }
        #steps.append(('est', est))
        #pipe = Pipeline(steps)
        #return pipe, param_dist
        return est, param_dist

    elif method_name in ['sofa', 'qsofa', 'sirs', 'mews', 'news']:
        from src.sklearn.baseline import BaselineModel
        import scipy.stats as stats
        parameters = {'column': method_name}
        parameters.update(clf_params)
        est = BaselineModel(**parameters)
        steps.append(('est', est))
        pipe = Pipeline(steps)
        param_dist = { 
            'est__theta':  stats.uniform(0, 1)
        }
        return pipe, param_dist 
    else:
        raise ValueError('Invalid method: {}'.format(method_name))


def main():
    """Parse arguments and launch fitting of model."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_path', default='datasets/{}/data/parquet/features_middle',
        help='Path to input data directory (relative from dataset directory)'
    )
    parser.add_argument(
        '--result_path', default='results',
        help='Relative path to experimental results (from input path)'
    )
    parser.add_argument(
        '--dataset', default='mimic_demo',
        help='Dataset Name: [physionet2019, ..]'
    )
    parser.add_argument(
        '--label_propagation', default=6, type=int,
        help="""(Active for classification task) By how many hours to 
            shift label into the past. Default: 6"""
    )
    parser.add_argument(
        '--label_propagation_right', default=24, type=int,
        help="""(Active for classification task) By how many hours to 
            shift label into the future, afterwards 0 again. Default: 24"""
    )
    parser.add_argument(
        '--overwrite', action='store_true', default=False,
        help='(Active for classification) To overwrite existing cached shifted labels'
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
        help='Number of iterations in randomized hyperparameter search'
    )
    parser.add_argument(
        '--cv_n_jobs', type=int, default=1,
        help='n_jobs for cross-validation'
    )
    parser.add_argument(
        '--variable_set', default='full',
        help="""which variable set should be used: [full, physionet], 
            where physionet refers to the subset as derived from 
            physionet challenge variables"""
    )
    parser.add_argument(
        '--feature_set', default='middle',
        help="""which feature set should be used: [small, middle], 
            large including feature engineering for classic models"""
    )
    parser.add_argument(
        '--split_path', 
        help='path to split file', 
        default='config/splits'
    )
    parser.add_argument(    
        '--normalizer_path', 
        help='path to normalization stats', 
        default='config/normalizer'
    )
    parser.add_argument(
        '--lambda_path', 
        help='path to lambda files', 
        default='config/lambdas'
    )
    parser.add_argument(
        '--feature_path', 
        help='path to feature names file', 
        default='config/features.json'
    )
    parser.add_argument(
        '--rep', 
        help='split repetition', type=int, 
        default=0)
    parser.add_argument(
        '--task', default='regression', 
        help='which prediction task to use: [classification, regression]'
    )
    parser.add_argument(
        '--index', default='multi',
        help='multi index vs single index (only pat_id, time becomes column): [multi, single]'
    )
    parser.add_argument(
        '--cost', default=0,
        type=int,
        help='lambda cost to use (default 0 (inactive))'
    )
    parser.add_argument(
        '--target_name', default='neg_log_loss',
        help='Only for classification: which objective to optimize in model selection [physionet_utility, roc_auc, average_precision]'
    )


    args = parser.parse_args()
    ## Process arguments:
    task = args.task 
 
    # Load data and current lambda and apply on-the-fly transforms:
    data, lam = load_data_splits(args)
    #data = load_data_from_input_path(
    #    args.input_path, args.dataset, args.index, args.extended_features)
    if task == 'classification':
        # for regression task the label shift happens in target calculation
        data = handle_label_shift(args, data)
 
    # TODO: add (update) baseline option! 
    ## for baselines: 
    #if args.method in ['sofa', 'qsofa', 'sirs', 'news', 'mews']:
    #    # use baselines as prediction input data
    #    # hack until prepro is run again (currently live jobs depending on it)
    #    data['baselines_train'].index = data['X_train'].index
    #    data['baselines_validation'].index = data['X_validation'].index
    #    data['X_train'] = data['baselines_train']
    #    data['X_validation'] = data['baselines_validation']
 
    pipeline, hparam_grid = get_pipeline_and_grid(args)
     
    if task == 'classification':
        target_name = args.target_name
        if target_name == 'physionet_utility':
            scores = {
                target_name: get_physionet2019_scorer(args.label_propagation,
                    kwargs={'lam': lam}    
                ),
                'roc_auc': SCORERS['roc_auc'],
                'average_precision': SCORERS['average_precision'],
                'balanced_accuracy': SCORERS['balanced_accuracy'],
            }
        elif target_name in ['roc_auc', 'average_precision', 'neg_log_loss']:
            scores = {
                'neg_log_loss': SCORERS['neg_log_loss'],
                'roc_auc': SCORERS['roc_auc'],
                'average_precision': SCORERS['average_precision'],
                #'balanced_accuracy': SCORERS['balanced_accuracy'],
            }
        else:
            raise ValueError(f'{target_name} not among valid targets.')
        print(f'Optimizing for {target_name}')
 
    elif task == 'regression':
        target_name = 'neg_mse'
        scores = { 
                target_name: SCORERS['neg_mean_squared_error'],
        }
     
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=hparam_grid,
        scoring=target_name,
        refit=target_name, #'average_precision'
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
    if task == 'regression': #only apply non-regression target scores here
        # since we have no sep3 label during random_search
        scores = { 
                target_name: SCORERS['neg_mean_squared_error'],
                'roc_auc': SCORERS['roc_auc'],
                'average_precision': SCORERS['average_precision'],
        }
    for score_name, scorer in scores.items():
        if task == 'regression' and score_name in ['roc_auc','average_precision']:
           results['val_' + score_name] = scorer._score(
                call, best_estimator, data['X_validation'].values, data['tp_labels_shifted_validation'].values)
        else: 
            results['val_' + score_name] = scorer._score(
                call, best_estimator, data['X_validation'].values, data['y_validation'].values)
    print(results)
    results['method'] = args.method
    results['best_params'] = random_search.best_params_
    results['n_iter_search'] = args.n_iter_search
    results['runtime'] = elapsed
    for method in ['predict', 'predict_proba', 'decision_function']:
        try:
            results['val_' + method] = call(
                best_estimator, method, data['X_validation'].values).tolist()
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
