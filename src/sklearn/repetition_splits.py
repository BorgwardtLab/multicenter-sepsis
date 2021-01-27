"""
This script is part of the preprocessing pipeline.

Here, we join the train/val split of the feature matrix and filtered matrix.
Then we create n repetition splits into train / val, in order to have 
repetitions splits to refit our models on - in order to report a performance
distribution on the test split.

"""

import os
import pickle
import argparse
import sys
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from src.sklearn.data.utils import load_pickle
from IPython import embed

def join_splits(path, splits):
    df = pd.DataFrame()
    for split in splits:
        data = load_pickle(
            path.format(split)
        )
        print(len(data)) 
        df = df.append(data)
    print(len(df))
    return df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path from dataset to pickled df file to inspect', 
        default='data/sklearn/processed')
    parser.add_argument('--dataset', type=str, help='dataset to use, when setting `all`, all datasets are iterated over', 
        default='physionet2019')
    parser.add_argument('--validation_size', type=float, default=0.111, 
     help='Ratio of patients in validation split, default 11.1% corresponds to 10% of entire dataset')
    parser.add_argument('--n_splits', type=int, default=5, help='how many repetition splits to create')
    
    # parse and unpack args
    args = parser.parse_args()
    dataset = args.dataset
    n_splits = args.n_splits
    val_size = args.validation_size

    # we join train and val split
    splits = ['train','validation']

    if dataset == 'all':
        datasets = ['demo', 'physionet2019', 'mimic3', 'eicu', 'hirid', 'aumc']
    else:
        datasets = [dataset]

    for dataset in datasets: 
        data = {}
        path = os.path.join('datasets', dataset, args.path)   
        
        # 1. Read, Join and Write extended features
        # features_path = os.path.join(path, 'X_extended_features_{}.pkl')
        # df = join_splits(features_path, splits) 
 
        # 2. Process filtered data (for deep models) 
        filtered_path = os.path.join(path, 'X_filtered_{}.pkl')
        df_f = join_splits(filtered_path, splits)
        baseline_path = os.path.join(path, 'baselines_{}.pkl')
        df_b = join_splits(baseline_path, splits)

        # 3. Create new splits
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=2323)
        
        ids = df_f.index.unique().tolist()
        labels = [df_f['sep3'].loc[id].any().astype(int) for id in ids]
 
        #for counter, (train_index, test_index) in enumerate(sss.split(y,y)):
        
        embed(); sys.exit()
       
  

