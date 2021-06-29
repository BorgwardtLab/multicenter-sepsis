import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import os
import pathlib
from time import time 
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

    def __call__(self, split='train', rep=0, test_repetitions=False):
        """
        Args:
        - split: which split to use 
                [train,validation,test]
        - rep: repetition to use [0 - 4]
            --> repetitions are by default only active 
                for train and validation split, 
                if repetitions of test split are to be used,
                set `test_repetitions=True`. Otherwise, for 
                the test split, the rep argument is ignored!
        """
        if split == 'test':
            if test_repetitions:
                ids = self.d[split][f'split_{rep}']
                print(f'{split} split repetition {rep} is used.')
            else:
                ids = self.d[split][f'total']
                print(f'The entire test split is used. Repetition argument ignored.')
        else:
            ids = self.d['dev'][f'split_{rep}'][split]
            print(f'{split} split repetition {rep} is used.')
        return ids 


class ParquetLoader:
    """Loads data from parquet by filtering for split ids."""
    def __init__(self, path, form='pandas', engine=None):
        """
        Arguments:
        - path: path to Parquet file (or folder)
        - form: format of returned table (pandas, dask, pyarrow)
        - engine: which engine to use for dask
            --> if loading specific ids `pyarrow-dataset` should be used,
            otherwise `pyarrow` which is robuster for general downstream 
            tasks - however when filtering for ids reads entire row groups 
            (which is typically not what we want here) 
        """
        self.path = path
        self.form = form
        self.engine = engine
        if engine is None and form == 'dask':
            self.engine = 'pyarrow-dataset'
        assert form in ['dask', 'pandas', 'pyarrow']
    
    def load(self, ids=None, filters=None, columns=None):
        """
        Args:
        - ids: which patient ids to load
        - filters: list of additional (optional) 
            filters: e.g. [('age', <, 70)]
        """
        filt = []
        if ids:
            filt = [ (VM_DEFAULT('id'), 'in', tuple(ids) ) ]
        if filters:
            filt.extend(filters)
        if len(filt) == 0:
            filt = None
        if self.form == 'dask':
            import dask.dataframe as dd
            print(f'Using dask with engine {self.engine}.')
            return dd.read_parquet(
                self.path, 
                filters=filt, 
                engine=self.engine, 
                columns=columns
            )
        elif self.form in ['pandas', 'pyarrow']:
            dataset = pq.ParquetDataset(
                self.path,
                use_legacy_dataset=False, 
                filters=filt
            )
            data = dataset.read(columns) if columns else dataset.read()
            if self.form == 'pandas':
                return data.to_pandas()
            else: 
                return data 


