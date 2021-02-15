"""
Here we assess different values of U_FP
"""

import argparse
import os
import pickle
import sys
from joblib import Parallel, delayed
import pandas as pd

from sklearn.pipeline import Pipeline

from functools import partial
from sklearn.metrics._scorer import _cached_call

from src.sklearn.data.transformers import *
from src.sklearn.data.utils import load_pickle, save_pickle, index_check
from src.evaluation.physionet2019_score import compute_prediction_utility, physionet2019_utility 
from src.evaluation import get_physionet2019_scorer
import seaborn as sns
from IPython import embed

class RandomBaseline:
    def __init__(self, r):
        self.r = r
    def fit(self, X,y):
        return self
    def predict(self, X):
        pred = np.random.binomial(1,self.r,(len(X)))
        return pred

def run_task(X, stats, i, n_draws, split, dataset, scorer):
    """
    task function for parallelism
    quadratic heuristic for U_FP
    """
    #prev = stats[stats['dataset'] == dataset]['tp_prevalence'].values[0]
    #d = {
    #    'hirid': -0.23,
    #    'physionet2019': -0.025,
    #    'eicu': -0.03,
    #    'mimic3': -0.075,
    #    'aumc': -0.045,
    #} 
    #u_fp = d[dataset] 
    utils, util_mu = get_average_util(X, i, n_draws=n_draws, scorer=scorer) 
    results = []
    for j, util in enumerate(utils):
        result = {}
        result['p'] = i
        result['split'] = split 
        result['dataset'] = dataset
        result['utility'] = util 
        result['draw'] = j
        results.append(result)
    return results

def get_average_util(X, r, n_draws=10, scorer=None):
    """
    get average util of data X with weight r and n bernoulli draws
    """
    #preds = np.random.binomial(1,r,(n_draws, len(X)))
    utils = []
    labels = X['sep3']
    for i in np.arange(n_draws):
        df = pd.DataFrame(labels)
        #df['pred'] = pred
        #ids = df.index.get_level_values('id').unique()
        #y_true = [df['sep3'].loc[id].values for id in ids]
        #y_pred = [df['pred'].loc[id].values for id in ids]
        est = RandomBaseline(r)
        cache = {}
        call = partial(_cached_call, cache)
        util = scorer._score(
            call, est, df, df) #the baseline simply draws bernoulli and needs the shape of the labels
        utils.append(util)
    return utils, np.mean(utils)  
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=str,
        help='Path from dataset to pickled df file to use as input',
        default='data/sklearn/processed'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset to use',
        default='demo'
    )
    parser.add_argument(
        '--split',
        type=str,
        help='Which split to use from [train, validation, test]; if not set '
             'we loop over all', 
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=10,
        help='parallel jobs'
    )

    args = parser.parse_args()
    split = args.split 
    dataset = args.dataset
    n_draws = 5

    if not split:
        splits = ['train', 'validation', 'test']
    else:
        splits = [split]
    if dataset == 'all':
        datasets = ['mimic3', 'physionet2019', 'hirid', 'eicu', 'aumc']
    else:
        datasets = [dataset]
    
    #old lambda: 
    #lam_dict = {'physionet2019': 0.9450719692139626, 
    #            'mimic3': 0.2264304193825734, 
    #            'hirid': 0.10039156674302935, 
    #            'aumc': 0.3726159954477741, 
    #            'eicu': 0.6234856778169942
    #}
    #new lambda without simplifying assumption:
    lam_dict = {'physionet2019': 1.0233822154472243, 'mimic3': 0.29985675708092846, 'hirid': 0.10233107525664238, 'aumc': 0.5474444498369464, 'eicu': 0.8577316443145103}
    
    out_path = os.path.join('results', 'pos_weight_analysis')
    stats = pd.read_csv(
        os.path.join(out_path, 'dataset_stats.csv')
    )

    #weights = np.arange(0,1.01,0.1)
    params =  np.arange(0,1.01,0.05)
    res_list = [] 
    for dataset in datasets:
        
        path = os.path.join('datasets', dataset, args.path)
         
        lam = lam_dict[dataset]
        scorer = get_physionet2019_scorer(shift=0, kwargs={'lam': lam}) #here we dont apply label propagation 

        print(f'Processing {dataset} and splits {splits}')
        for split in splits: 
            name = f'X_features_{split}'
            features_path = os.path.join(path, name + '.pkl')
            X = load_pickle(features_path)
            X = index_check(X)
            
            res = Parallel(n_jobs=args.n_jobs)(
                delayed(run_task)(  X, 
                                    stats, 
                                    i, 
                                    n_draws, 
                                    split, 
                                    dataset,
                                    scorer) for i in params)
            res_list.append(res)
             
    # unpack parallelized results to dict of lists:
    results = {}
    keys = ['p', 'split', 'dataset', 'utility', 'draw']
    for key in keys:
        results[key] = [] 
    
    for res in res_list: # over datasets and splits
        for elem in res: # list of parallelized outputs
            for d in elem: # each output is a list of dicts
                for key in keys:
                    results[key].append(d[key])
    
    df = pd.DataFrame.from_dict(results)
    
    os.makedirs(out_path, exist_ok=True)
    df.to_csv(
        os.path.join(out_path, 'random_baselines_parallel_dictionary_lambda2.csv')
    ) 
    
         
        
        
