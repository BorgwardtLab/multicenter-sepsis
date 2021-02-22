import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json
import os

from src.variables.mapping import VariableMapping as VM

class SplitInfo:
    """
    Class that handles split information and 
    returns patient ids.
    """
    def __init__(self, split_path):
        """
        Args:
        - split_path: path to file containing split information
        """
        self.split_path = split_path
        self.d = self._load_info(split_path)

    def _load_info(self, path):
        with open(path, 'r') as f:
            return json.load(f) 

    def __call__(self, split='train', rep=0):
        """
        Args:
        - split: which split to use 
                [train,validation,test]
        - rep: repetition to use [0 - 4]
        """
        if split == 'test':
            ids = self.d[split][f'split_{rep}']
        else:
            ids = self.d['dev'][f'split_{rep}'][split]
        return ids 


class ParquetLoader:
    """Loads data from parquet by filtering for split ids."""
    def __init__(self, path):
        self.path = path
        self.vm = VM() # get variable mapping
    
    def load(self, ids, filters=[], pandas=True):
        """
        Args:
        - ids: which patient ids to load
        - filters: list of additional (optional) 
            filters: e.g. [('age', <, 70)]
        - pandas: flag if pd.DataFrame should 
            be returned
        """
        filt = [ (self.vm('id'), 'in', ids ) ]
        if len(filters) > 0:
            filt.extend(filters)
        
        dataset = pq.ParquetDataset(
            self.path,
            use_legacy_dataset=False, 
            filters=filt
        )
        if pandas:
            return dataset.read().to_pandas()
        else: 
            return dataset.read()

 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
                        help='name of dataset to use', 
                        default='demo')
    parser.add_argument('--split_path', 
                        help='path to split file', 
                        default='config/splits')
    parser.add_argument('--split', 
                        help='which data split to use', 
                        default='train')
    parser.add_argument('--rep', 
                        help='split repetition', type=int, 
                        default=0)
    args = parser.parse_args()
    dataset_name = args.dataset 
    path = f'datasets/{dataset_name}/data/parquet/features.parquet'
    split_path = os.path.join(args.split_path, f'splits_{dataset_name}.json' ) 
    split = args.split 
    rep = args.rep

    si = SplitInfo(split_path)
    ids = si(split, rep)
    f1 = ('age', '<', 70) 
     
    pl = ParquetLoader(path)
    df = pl.load(ids)
    df2 = pl.load(ids, [f1])
    t = pl.load(ids,pandas=False) 
    from IPython import embed; embed()