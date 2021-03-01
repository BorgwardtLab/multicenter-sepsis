import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json
import os
import pathlib

from src.preprocessing.transforms import Normalizer

from src.variables.mapping import VariableMapping

VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)
VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


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
    
    def load(self, ids=None, filters=None, columns=None, pandas=True):
        """
        Args:
        - ids: which patient ids to load
        - filters: list of additional (optional) 
            filters: e.g. [('age', <, 70)]
        - pandas: flag if pd.DataFrame should 
            be returned
        """
        filt = []
        if ids:
            filt = [ (VM_DEFAULT('id'), 'in', ids ) ]
        if filters:
            filt.extend(filters)
        if filt:
            dataset = pq.ParquetDataset(
                self.path,
                use_legacy_dataset=False, 
                filters=filt
            )
        else:
            dataset = pq.ParquetDataset(
                self.path,
                use_legacy_dataset=False, 
            )
 
        data = dataset.read(columns) if columns else dataset.read()
        if pandas:
            return data.to_pandas()
        else: 
            return data 

def load_and_transform_data(
    data_path,
    split_path,
    normalizer_path,
    lambda_path,
    split='train',
    rep=0,
):
    """
    Data loading function (for classic models).
    Applies on the fly transforms:
        1. read patient ids according to split
        2. apply precomputed normalization 
    """
    # determine patient ids of current split:
    si = SplitInfo(split_path)
    ids = si(split, rep)
    # load these patient ids:
    pl = ParquetLoader(data_path)
    df = pl.load(ids)

    # set up normalizer:
    ind_cols = [col for col in df.columns if '_indicator' in col]
    drop_cols = [
        VM_DEFAULT('label'),
        VM_DEFAULT('sex'),
        VM_DEFAULT('time'),
        VM_DEFAULT('utility'),
        *VM_DEFAULT.all_cat('baseline'),
        *ind_cols
    ]
 
    norm = Normalizer(patient_ids=None, drop_cols=drop_cols)
    with open(normalizer_path, 'r') as f:
        normalizer = json.load(f)
    # we load and set the statistics
    norm.stats = {
            'means': pd.Series(normalizer['means']),
            'stds': pd.Series(normalizer['stds'])
        }
    df_norm = norm.transform(df)
    
    # We apply lambda sample weights:
    with open(lambda_path, 'r') as f:
        lam = json.load(f)['lam']
    # regression target without adjustment:
    u = df_norm[VM_DEFAULT('utility')]
    # patient-level label:
    l = df.groupby('stay_id')['sep3'].sum() > 0 #more nan-stable than .any()
    # applying lambda to target: if case: times lam, else no change
    df_norm[VM_DEFAULT('utility')] = l*u*lam + (1-l)*u 

    return df_norm
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
                        help='name of dataset to use', 
                        default='demo')
    parser.add_argument('--split_path', 
                        help='path to split file', 
                        default='config/splits')
    parser.add_argument('--normalizer_path', 
                        help='path to normalization stats', 
                        default='config/normalizer')
    parser.add_argument('--lambda_path', 
                        help='path to lambda file', 
                        default='config/lambdas')
    parser.add_argument('--split', 
                        help='which data split to use', 
                        default='train')
    parser.add_argument('--rep', 
                        help='split repetition', type=int, 
                        default=0)
    args = parser.parse_args()
    dataset_name = args.dataset 
    path = f'datasets/{dataset_name}/data/parquet/features' 
    split_path = os.path.join(args.split_path, 
        f'splits_{dataset_name}.json' ) 
    split = args.split 
    rep = args.rep
    normalizer_path = os.path.join(args.normalizer_path, 
        f'normalizer_{dataset_name}_rep_{rep}.json' )
    lambda_path = os.path.join(args.lambda_path, 
        f'lambda_{dataset_name}_rep_{rep}.json' ) 
    
    load_and_transform_data(
    path,
    split_path,
    normalizer_path,
    lambda_path,
    split,
    rep)


