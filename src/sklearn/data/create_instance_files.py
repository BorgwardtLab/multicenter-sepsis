import os
import pickle
import argparse
import sys
sys.path.append(os.getcwd())

from src.sklearn.data.utils import load_pickle, save_pickle
from IPython import embed
import pandas as pd


class InstanceWriter():
    def __init__(self, path):
        self.path = path
 
    def _write_instance(self, df):
        fname = str(df.index[0]) + '.pkl' 
        fpath = os.path.join(self.path, fname)
        save_pickle(df, fpath) 

    def to_instance_files(self, df):
        df = df.groupby('id', ).apply(self._write_instance) #group_keys=False

    def from_instance_files(self, filepath):
        """
        sanity check, that it works properly
        """
        files = os.listdir(filepath)
        files = [f for f in files if ('.pkl' in f) and not ('info' in f) ]
        result = pd.DataFrame()
        for f in files:
            df = load_pickle(os.path.join(filepath, f))
            result = result.append(df)
        #properly sort instances and times after loading from separate files
        result = result.reset_index().sort_values(['id', 'time']).set_index('id')
        return result
         
        
 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path from dataset to pickled df file to use as input', 
        default='data/sklearn/processed')
    parser.add_argument('--dataset', type=str, help='dataset to use', 
        default='demo')
    parser.add_argument('--split', type=str, help='which split to use from [train, validation, test], if not set we loop over all', 
        default=None)
    
    args = parser.parse_args()
    split = args.split 
    dataset = args.dataset 
    path = os.path.join('datasets', dataset, args.path)
     
    if split is None:
        splits = ['train', 'validation', 'test']
    else:
        splits = [split]
    print(f'Processind {dataset} and splits {splits}')

    for split in splits: 
        # name of the input pickle file 
        name = f'X_features_{split}'
        features_path = os.path.join(path, name + '.pkl')
        X = load_pickle(features_path)
      
        # outpath for the instance files 
        outpath = os.path.join(path, 'instances', name) 
        iw = InstanceWriter(outpath)
        iw.to_instance_files(X)
        
        # if demo dataset is used, reconstruct the df and check that it is the same:
        if dataset == 'demo': 
            X_r = iw.from_instance_files(outpath)
            try: 
                assert X.equals(X_r)
            except:
                print('assert failed')
                embed() 
    
        # add small df of all patients for easier down-stream handling (e.g. of additional splitting for hyperparam tuning)
        # including id and label
        info_file = os.path.join(outpath, 'info.pkl')
        info_df = X[['time', 'sep3']]
        save_pickle(info_df, info_file) 
