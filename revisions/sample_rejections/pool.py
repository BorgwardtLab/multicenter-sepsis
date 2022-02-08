# Consider uncertainty band / rejection for the pooled predictions:

import os
from glob import glob
import pandas as pd
from pathlib import Path
import numpy as np
import json
import argparse
from IPython import embed
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import average_precision_score as auprc 

def read_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def write_json(obj, fname):
    with open(fname, 'w') as f:
        json.dump(obj, f, indent=4)

data_mapping = {
        'mimic': 'MIMIC',
        'hirid': 'Hirid',
        'eicu': 'EICU',
        'aumc': 'AUMC',
        'pooled': 'pooled',
    }

def format_dataset(x):
    if x in data_mapping.keys():
        x = data_mapping[x]
    return x

def isig(x):
    """ inverse sigmoid, to map probs to logits"""
    return np.log(x/ (1-x))

def sig(x):
    """ sigmoid """
    return (1 / (1 + np.exp(-x)))


def main(args):
    if not args.load:
        files = glob(args.input + '/*.csv')

        current_files = list(filter(lambda x: args.dataset_eval in x, files))
        # used models, and whether they use probabilities:
        models = {'AttentionModel': False, 
                  'GRUModel': False, 
                  'lgbm': True
        }
        all_datasets = ['MIMIC', 'Hirid', 'AUMC', 'EICU']
        # currently used training datasets:
        current_datasets = [d for d in all_datasets if d != data_mapping[args.dataset_eval]]
        
        current_files = [f for f in current_files for m in models.keys() if m in f]
        df_ = pd.DataFrame()

        for f in current_files:
            model = f.split('_')[-2] # strong assumption on format of f!
            assert model in models.keys()
            df = pd.read_csv(f)
            # make sure that datasets are formatted uniquely:
            df = df.rename(columns = data_mapping) 
            # also eval_dataset col needs to be formatted uniquely:
            dataset_eval = df['dataset_eval'][0]
            if dataset_eval in data_mapping.keys():
                df['dataset_eval'] = data_mapping[dataset_eval] 

            # sort values of df properly:
            total_df = []
            for (rep, subsample), curr_df in df.groupby(['rep','subsample']):
                curr_df = curr_df.sort_values(['stay_id', 'stay_time'])
                total_df.append(curr_df)
            df = pd.concat(total_df) 
            
            if not models[model]: #if don't have probabilities
                df[current_datasets] = sig(df[current_datasets])
            
            # add model suffix to dataset specific model prediction columns:
            df = pd.concat([df, df[current_datasets].add_suffix(f'_{model}')], axis=1)
            df.drop(columns=current_datasets, inplace=True) 
            model_cols = [c for c in df.columns if model in c]

            index_cols = ['stay_id', 'stay_time', 'labels', 'rep', 'subsample', 'dataset_eval']
            assert all([c in df.columns for c in index_cols])
            
            df = df.set_index(index_cols)

            if len(df_) == 0:
                df_ = df.copy()
            else:
                #check that multi-index matches:
                try:
                    assert all(df_.index == df.index)
                except:
                    print(f'Index did not match!')
                    embed()
                df_[model_cols] = df[model_cols] 

        output_file = os.path.join(args.input, 'collected')
        os.makedirs(output_file, exist_ok=True)
        models = '_'.join(models.keys())
        output_file = os.path.join(output_file, 
            f'collected_scores_{args.dataset_eval}_{models}.csv' 
        )
        df_ = df_.reset_index() # was only used to check that index is set properly
        df_.to_csv(output_file, index=False)
    else:
        input_file = glob(
            os.path.join(args.input, 'collected', 
                f'collected_scores_{args.dataset_eval}_*.csv' 
            )
        )[0]
        print(f'Loading file: {input_file}')
        df_ = pd.read_csv(input_file)
        embed()
        
    # consider one rep and subsample:
    stats = ['max', 'mean', 'median', 'min', 'std']
    averages = {s: [] for s in stats}
    for (rep,subsample), df in df_.groupby(['rep', 'subsample']):
        #dfg = df_.groupby(['rep', 'subsample'])
        #df = dfg.get_group((0,0))    
        labels = df.groupby('stay_id')['labels'].max() 
       
        score_cols = df.columns[6:] # columns of prediction scores 
        score_cols = [s for s in score_cols if 'AttentionModel' in s]
        
        for stat in stats:
            df[stat] = getattr(df[score_cols], stat)(axis=1)
        # aggregate over time points per patient:
        score_stats = df.groupby('stay_id')[stats].mean() #max()
        embed()

        # Test masking:
        lower = np.percentile(score_stats['mean'], 40)
        upper = np.percentile(score_stats['mean'], 60)
        print(f'Lower = {lower}, upper = {upper}')   
 
        mask = (score_stats['mean'] > lower ) * (score_stats['mean'] < upper)
 
        score_stats = score_stats[~mask]
        labels = labels[~mask] 

        for stat in stats: 
            auc = roc_auc(labels, score_stats[stat])
            aupr = auprc(labels, score_stats[stat])
            # print(f'Rep {rep}, Subsample {subsample}, {stat} AUC = {auc}')
            averages[stat].append(auc)
            #averages[stat].append(aupr)
    for k in averages.keys():
        print(f'{k}, average AUC = {np.mean(averages[k])}') 
        #print(f'{k}, average AUPRC = {np.mean(averages[k])}') 

    embed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default = 'results/evaluation_test/prediction_pooled_subsampled', 
        help = 'input path to raw result file'
    )
    parser.add_argument(
        '--output_dir',
        default = 'results/sample_rejections', 
        help = 'path to output folder'
    )
    parser.add_argument(
        '--lower',
        default = 40, type=int, 
        help = 'lower end of percentile for uncertainty / rejection mask'
    )
    parser.add_argument(
        '--upper',
        default = 60, type=int, 
        help = 'upper end of percentile for uncertainty / rejection mask'
    )
    parser.add_argument(
        '--dataset_eval',
        default='eicu',
        help = 'dataset to evaluate pooled scores on'
    )
    parser.add_argument(
        '--load',
        action='store_true', 
        help = 'Flag to load preprocessed file (quicker for development)'
    )
 

    args = parser.parse_args()
    main(args)


