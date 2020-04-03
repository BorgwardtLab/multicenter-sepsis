import argparse 
from collections import defaultdict
import numpy as np
import os
import pandas as pd
from IPython import embed
import sys
from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.externals import joblib
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

sys.path.append(os.getcwd())
from src.sklearn.data.utils import load_data, save_pickle

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', 
                        default='data/sklearn', 
                        help='Path to input data directory (relative from dataset directory)')
    parser.add_argument('--result_path', 
                        default='results',
                        help='Relative path to experimental results (from input path)')
    parser.add_argument('--dataset', 
                        default='physionet2019',
                        help='Dataset Name: [physionet2019, ..]')
    parser.add_argument('--overwrite', 
                        action='store_true', default=True,
                        help='<Currently inactive> To overwrite existing preprocessed files')
    parser.add_argument('--method', 
                        default='lgbm', type=str, 
                        help='<Method to use for classification [lgbm, ..]')
    parser.add_argument('--n_iter_search', 
                        type=int, 
                        default=20,
                        help='Number of iterations in randomized search for hyperparameter optimization')
    parser.add_argument('--clf_n_jobs', 
                        type=int, 
                        default=10,
                        help='n_jobs for classifier model (if method allows it!)')
    parser.add_argument('--cv_n_jobs', 
                        type=int, 
                        default=10,
                        help='n_jobs for cross-validation')
    
    # Parse Arguments and unpack the args:
    args = parser.parse_args()
    dataset = args.dataset
    input_path = os.path.join('datasets', dataset, args.input_path) # path to dataset 
    result_path = os.path.join(input_path, args.result_path)
    os.makedirs(result_path, exist_ok=True)
    
    overwrite = args.overwrite
    n_iter_search = args.n_iter_search
    method = args.method

    ##Check if the classification ran before succesfully (result file exists)
    #result_file = os.path.join(args.result_path, method + '_' + dataset + '.csv') 
    #if os.path.exists(result_file) and not overwrite:
    #    print('Classification Result already exists. Skip this job..')
    #    sys.exit()
    
    # load data 
    data  = load_data(path=os.path.join(input_path, 'processed'))
    # unpack train and validation splits:
    X_train, y_train = data['X_train'].values, data['y_train'].values
    X_val, y_val = data['X_validation'].values, data['y_validation'].values

    ######
    if method == 'lgbm':
        import lightgbm as lgb
        est = lgb.LGBMClassifier(n_jobs=args.clf_n_jobs)
        pipe = Pipeline(steps=[('est', est)])
        param_dist = {
        'est__n_estimators': [50,100,300,500, 1000],
        'est__boosting_type': ['gbdt', 'dart'],
        'est__learning_rate':[0.001, 0.01, 0.1, 0.5],
        'est__num_leaves': [30, 50, 100],
        'est__scale_pos_weight': [1,10,20,50,100]
        }
    else:
        raise ValueError(f'Provided Method {method} not implemented! Choose from [lgbm]')
    #Use defined pipeline and param_dist for randomized search:
    
    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, scoring='roc_auc', #'average_precision',
                                   n_iter=n_iter_search, cv=5, iid=False, n_jobs=args.cv_n_jobs)
    start = time()
    rs.fit(X_train, y_train)
    elapsed = time() - start 
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((elapsed), n_iter_search))

    y_pred = rs.best_estimator_.predict(X_val)
    #save predictions (for utility score)
    pred_path = os.path.join(input_path, 'predictions')
    os.makedirs(pred_path, exist_ok=True)
    save_pickle(y_pred, os.path.join(pred_path, 'y_pred.pkl')) 
    
    avp = average_precision_score(y_val, y_pred)
    print(f'Val AVP: {avp}')
    #Write results to csv:
    results = defaultdict() 
    results['val_avp'] = [avp]
    results['method'] = [method]
    results['n_iter_search'] = [n_iter_search]
    results['runtime'] = [elapsed]
 
    df = pd.DataFrame.from_dict(results, orient='columns')
    print(df)
    result_file = os.path.join(result_path, method +'.csv') 
    df.to_csv(result_file,index=False)    
    #also dump best estimator to result path:
    joblib.dump(rs.best_estimator_, os.path.join(result_path, 'best_estimator.pkl'),
         compress = 1)    
 
if __name__ in "__main__":
    
    main()

