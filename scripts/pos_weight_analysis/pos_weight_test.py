import argparse
import os
import pickle
import sys

import pandas as pd

from sklearn.pipeline import Pipeline

from src.sklearn.data.transformers import *
from src.sklearn.data.utils import load_pickle, save_pickle, index_check
from src.evaluation.physionet2019_score import compute_prediction_utility, physionet2019_utility 

import seaborn as sns

def get_average_util(X, r, n_draws=10):
    """
    get average util of data X with weight r and n bernoulli draws
    """
    preds = np.random.binomial(1,r,(n_draws, len(X)))
    utils = []
    labels = X['sep3']
    for i, pred in enumerate(preds):
        df = pd.DataFrame(labels)
        df['pred'] = pred
        ids = df.index.get_level_values('id').unique()
        y_true = [df['sep3'].loc[id].values for id in ids]
        y_pred = [df['pred'].loc[id].values for id in ids] 
        util = physionet2019_utility(y_true, y_pred)
        #util = compute_prediction_utility(
        #    labels,
        #    pred,
        #    shift_labels=0,
        #    return_all_scores=False
        #)
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
    
    results = {}
    results['p'] = []
    results['split'] = []
    results['dataset'] = []
    results['r_soft'] = []
    results['r_hard'] = []
    results['utility'] = []
    results['draw'] = []
  
    for dataset in datasets:
        
        path = os.path.join('datasets', dataset, args.path)
         
        data_pipeline = Pipeline([
            ('calculate_utility_scores',
                CalculateUtilityScores(passthrough=False, n_jobs=20))]
        )

        print(f'Processing {dataset} and splits {splits}')

        for split in splits: 
            name = f'X_features_{split}'
            features_path = os.path.join(path, name + '.pkl')
            X = load_pickle(features_path)
            X = index_check(X)
            X = data_pipeline.fit_transform(X)
            
            u = X['utility']
            s_p = np.sum(u[ u > 0])
            s_n = np.sum(np.abs(u[ u < 0])) # zeros don't matter 
            r_soft = s_p / (s_p + s_n)
            
            s_p = len(u[ u > 0])
            s_n = len(u[ u < 0])    
            r_hard = s_p / (s_p + s_n) 
            print(f'Optimal r_soft = {r_soft}')
            print(f'Optimal r_hard = {r_hard}')

            #u, u_mu = get_average_util(X, r=1, n_draws=1) #r=r
             
            for i in np.arange(0,1.01,0.05):
                utils, util_mu = get_average_util(X, i, n_draws=n_draws)
                print(f'p = {i:.3f}', f'util = {util_mu:.3f}')
                
                for j, util in enumerate(utils):
                    results['p'].append(i)
                    results['split'].append(split)
                    results['dataset'].append(dataset)
                    results['r_soft'].append(r_soft)
                    results['r_hard'].append(r_hard)
                    results['utility'].append(util)
                    results['draw'].append(j) 
    
    df = pd.DataFrame.from_dict(results)
    out_path = os.path.join('results', 'pos_weight_analysis')
    os.makedirs(out_path, exist_ok=True)
    df.to_csv(
        os.path.join(out_path, 'random_baselines.csv')
    ) 
    
         
        
        
