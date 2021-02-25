import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json

from .mapping import VariableMapping as VM
from src.sklearn.loading import SplitInfo, ParquetLoader

def extract_feature_cats(cols):
    """ util function to extract feature tags """
    columns = []
    vm = VM()
    for col in cols:
        # feature categories are indicated as _featuregroup
        s = col.split('_') 
        if len(s) == 1: # variable without category 
            if col not in vm.all_cat('baseline'):
                columns.append(col)
        else:
            prefix = '_' + s[1] # feature cat is second entry
        if prefix not in columns:
            columns.append(prefix)

    
    return sorted(columns) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
                        help='name of dataset to use', 
                        default='demo')
    parser.add_argument('--split_path', 
                        help='path to split file', 
                        default='config/splits/splits_{}.json')
    parser.add_argument('--split', 
                        help='which data split to use', 
                        default='train')
    parser.add_argument('--rep', 
                        help='split repetition', type=int, 
                        default=0)
    args = parser.parse_args()
    dataset_name = args.dataset 
    path = f'datasets/{dataset_name}/data/parquet/features.parquet'
    split_path = args.split_path.format(dataset_name) 
    split = args.split 
    rep = args.rep

    si = SplitInfo(split_path)
    ids = si(split, rep)
     
    pl = ParquetLoader(path)
    t = pl.load(ids,pandas=False)

    columns = extract_feature_cats(t.column_names)
 
    from IPython import embed; embed()
