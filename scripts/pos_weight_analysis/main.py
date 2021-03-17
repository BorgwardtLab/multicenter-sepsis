"""
Here we assess different values of U_FP
"""

import argparse
import os
import json
import sys
from joblib import Parallel, delayed
import pandas as pd

from sklearn.pipeline import Pipeline
from time import time
import numpy as np 
from functools import partial
from sklearn.metrics._scorer import _cached_call

from src.sklearn.loading import SplitInfo, ParquetLoader
#from src.sklearn.data.transformers import *
from src.sklearn.data.utils import index_check
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

def run_task(X, i, n_draws, split, dataset, scorer):
    """
    task function for parallelism
    evaluate utility for a given threshold 
    """
    
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
        help='Generic path to dataset features (keeping dataset name abstract {})',
        default='datasets/{}/data/parquet/features'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset to use',
        default='mimic_demo'
    )
    parser.add_argument(
        '--split',
        type=str,
        help='Which split to use from [train, validation, test]; if not set '
             'we loop over all', 
    )
    parser.add_argument(
        '--split_path', 
        help='path to split file', 
        default='config/splits'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=10,
        help='parallel jobs'
    )
    parser.add_argument('--lambda_path', 
                        help='path to lambda file', 
                        default='config/lambdas')
    parser.add_argument('--ignore_lam', 
                        help='flag to ignore lambda',
                        action='store_true' 
    )
    args = parser.parse_args()
    split = args.split 
    dataset = args.dataset
    
    # hard-coded parameters:
    n_draws = 5
    rep = 0
    

    if not split:
        splits = ['train', 'validation', 'test']
    else:
        splits = [split]
    if dataset == 'all':
        datasets = ['mimic', 'physionet2019', 'hirid', 'eicu', 'aumc']
    else:
        datasets = [dataset]
    
    #simplified lambda: 
    #lam_dict = {'physionet2019': 0.9450719692139626, 
    #            'mimic3': 0.2264304193825734, 
    #            'hirid': 0.10039156674302935, 
    #            'aumc': 0.3726159954477741, 
    #            'eicu': 0.6234856778169942
    #}
    #new lambda without simplifying assumption:
    # lam_dict = {'physionet2019': 1.0233822154472243, 'mimic3': 0.29985675708092846, 'hirid': 0.10233107525664238, 'aumc': 0.5474444498369464, 'eicu': 0.8577316443145103}
    # lambda without simplifying assumption on updated data (label, filtering update), validation split, rep 0
     
    out_path = os.path.join('results', 'pos_weight_analysis')
    
    #weights = np.arange(0,1.01,0.1)
    params =  np.arange(0,1.01,0.05)
    res_list = [] 
    for dataset in datasets:
        # Determine current lambda:
        lambda_path = os.path.join(args.lambda_path, 
            f'lambda_{dataset}_rep_{rep}.json' )
        if args.ignore_lam:
            lam = 1.0
        else:
            with open(lambda_path, 'r') as f:
                lam = json.load(f)['lam']
        # Initialize split and dataset classes: 
        split_path = os.path.join(args.split_path, 
        f'splits_{dataset}.json' ) 
        si = SplitInfo(split_path)
         
        # Load Patient Data (selected ids and columns):
        data_path = args.path.format(dataset)
        pl = ParquetLoader(data_path, form='pandas')
        
        #lam = lam_dict[dataset]
        print(f'Using lambda: {lam}.')
        scorer = get_physionet2019_scorer(shift=0, kwargs={'lam': lam}) #here we dont apply label propagation 

        print(f'Processing {dataset} and splits {splits}')
        for split in splits:
            # Load current split:
            ids = si(split, rep)
            start = time()
            print(f'Loading patient data..')
            df = pl.load(ids, columns=['stay_id','stay_time','sep3'])
            print(f'.. took {time() - start} seconds.') 
            
            print(f'Ensuring multi-index') 
            df = index_check(df)
            #name = f'X_features_{split}'
            #features_path = os.path.join(path, name + '.pkl')
            #X = load_pickle(features_path)
            #X = index_check(X)
            
            res = Parallel(n_jobs=args.n_jobs)(
                delayed(run_task)(  df, 
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
    out_path = os.path.join(out_path, 'random_baselines_parquet_{}.csv')
    if args.ignore_lam:
        out_path = out_path.format('no_lam')
    else:
        out_path = out_path.format('')
    df.to_csv(
        out_path
    ) 
