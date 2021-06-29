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


def load_and_transform_data(
    data_path,
    split_path,
    normalizer_path,
    lambda_path,
    feature_path,
    split='train',
    rep=0,
    feature_set='middle',
    variable_set='full',
    task='regression',
    baselines=False,
    form='pandas',
    test_repetitions=False
):
    """
    Data loading function (for classic models).
    Applies on the fly transforms:
        1. read patient ids according to split
        2. apply precomputed normalization 
    Arguments:
    - data_path: path to preprocessed data (as parquet dumps)
    - split_path: path to split json file
    - normalizer_Path: path to normalization json file
    - lambda_path: path to lambda (sample weight) json file 
    - feature_path: path to feature names json file 
    - split: [train, validation, test]
    - rep: reptition [0 to 4]
    - feature_set: middle or small (classic or deep models)
    - variable_set: full or physionet (fewer variables)
    - task: regression or classification
    - baselines: flag, if set overrides feature and variable set
        and only loads baselines as input features
    - form: format to return, (pandas,dask)
    - test_repetitions: flag to return boosted test repetitions, 
        otherwise full test set returned if split='test' and rep only 
        used for normalization.

    returns data and lambda 
    """
    print(f'Loading {feature_set} feature set')
    # determine columns to load:
    with open(feature_path, 'r') as f:
        feat_dict = json.load(f)
    if baselines:
        cols = VM_DEFAULT.all_cat('baseline')
        cols.extend([VM_DEFAULT(x) for x in ['id', 'time']])
    else:
        if feature_set in feat_dict[variable_set].keys(): 
            cols = feat_dict[variable_set][feature_set]['columns']
        else:
            print(f'Custom feature set {feature_set} not among cached feature sets, computing it on the fly, but ONLY FOR THE FULL variable set!')
             
    if task == 'regression':
        cols.extend([VM_DEFAULT(x) for x in ['label', 'utility']]) 
        # as we use label in lambda application
    elif task == 'classification':
        cols.append(VM_DEFAULT('label'))
    else: 
        raise ValueError(f'task {task} not among valid tasks: regression, classification') 
    if form == 'dask':
        cols.remove(VM_DEFAULT('id')) #already in index

    # determine patient ids of current split:
    si = SplitInfo(split_path)
    ids = si(split, rep, test_repetitions)

    # 1. Load Patient Data (selected ids and columns):
    pl = ParquetLoader(data_path, form=form)
    start = time()
    print(f'Loading patient data..')
    df = pl.load(ids, columns=cols)
    print(f'.. took {time() - start} seconds.')
    # 2. Apply Normalization (if we are not loading baseline scores)
    if not baselines: # we only need normalization on the real input features
        # set up normalizer:
        ind_cols = [col for col in df.columns if '_indicator' in col]
        drop_cols = [
            VM_DEFAULT('label'),
            VM_DEFAULT('sex'),
            VM_DEFAULT('time'),
            VM_DEFAULT('utility'),
            *VM_DEFAULT.all_cat('baseline'),
            *ind_cols,
            VM_DEFAULT('id') # for dask in index
        ]
        start = time()
        print('Applying normalization..') 
        norm = Normalizer(patient_ids=None, drop_cols=drop_cols)
        with open(normalizer_path, 'r') as f:
            normalizer = json.load(f)
        # we load and set the statistics
        means = pd.Series(normalizer['means'])
        stds = pd.Series(normalizer['stds'])
        # actually used columns for normalization:
        norm_cols = list(set(means.index).intersection(cols))
        norm.stats = {
                'means': means.loc[norm_cols],
                'stds': stds.loc[norm_cols]
            }
        df = norm.transform(df)
        print(f'.. took {time() - start} seconds.')

    # 3. Load and apply lambda sample weight if we use regression target:
    with open(lambda_path, 'r') as f:
        lam = json.load(f)['lam']
    if task == 'regression':
        # For regression, we apply lambda sample weights and remove label
        start = time()
        print(f'Applying lambda {lam}..')
        # regression target without adjustment:
        u = df[VM_DEFAULT('utility')]
        # patient-level label:
        timestep_label = df[VM_DEFAULT('label')]
        l = timestep_label.groupby(VM_DEFAULT('id')).sum() > 0 #more nan-stable than .any()
        l = l.astype(int)
        l = l.reindex(u.index) # again timestep wise label
        # applying lambda to target: if case: times lam, else no change
        # need to drop target instead of overwriting as otherwise pandas 
        # throwed an error due to reindexing with duplicate indices..
        df = df.drop(columns=[VM_DEFAULT('utility')])
        new_target = l*u*lam + (1-l)*u 
        df[VM_DEFAULT('utility')] = new_target 
        # df = df.drop(columns=[VM_DEFAULT('label')]) 
        print(f'.. took {time() - start} seconds.')

    # 4. Remove remaining NaN values and check for invalid values 
    # e.g. due to degenerate stats in normalization
    start = time()
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0) #sanity check if there is degenerate normalization
    print(f'Final imputation took {time() - start} seconds.')
    if form == 'dask':
        df = df.compute()
    return df, lam
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
                        help='name of dataset to use', 
                        default='mimic_demo')
    parser.add_argument('--split_path', 
                        help='path to split file', 
                        default='config/splits')
    parser.add_argument('--normalizer_path', 
                        help='path to normalization stats', 
                        default='config/normalizer')
    parser.add_argument('--lambda_path', 
                        help='path to lambda file', 
                        default='config/lambdas')
    parser.add_argument('--feature_path', 
                        help='path to feature names file', 
                        default='config/features.json')
    parser.add_argument('--split', 
                        help='which data split to use', 
                        default='train')
    parser.add_argument('--rep', 
                        help='split repetition', type=int, 
                        default=0)
    parser.add_argument('--dump_name', 
                        help='which feature dump to use',
                        default='features')
    parser.add_argument('--cache_path', 
                        help="""if provided transformed df is cached as parquet to this base_dir 
                        in a folder {dump_name}_cache/{split}_{rep}""", 
                        default=None)
    parser.add_argument('--feature_set', 
                        help='which feature set to use (middle, small)', 
                        default='small')
    parser.add_argument('--cost', type=int, 
                        help='cost parameter in lambda to use',
                        default=0)

    args = parser.parse_args()
    dataset_name = args.dataset
    dump_name = args.dump_name 
    path = f'datasets/{dataset_name}/data/parquet/{dump_name}' 
    split_path = os.path.join(args.split_path, 
        f'splits_{dataset_name}.json' ) 
    split = args.split 
    rep = args.rep
    cost = args.cost
    normalizer_path = os.path.join(args.normalizer_path, 
        f'normalizer_{dataset_name}_rep_{rep}.json' )
    if cost > 0:
        lam_file = f'lambda_{dataset_name}_rep_{rep}_cost_{cost}.json'
    else:
        lam_file = f'lambda_{dataset_name}_rep_{rep}.json'
    lambda_path = os.path.join(args.lambda_path, 
        lam_file )
 
    df, _ = load_and_transform_data(
        path,
        split_path,
        normalizer_path,
        lambda_path,
        args.feature_path,
        split,
        rep,
        feature_set=args.feature_set,
        variable_set='full',
        task='regression',
        baselines=False
    )
    cache_path = args.cache_path
    if cache_path:
        
        cache_path = os.path.join(
            cache_path,
            f'{dump_name}_cache'
        )
        os.makedirs(cache_path, exist_ok=True)
        cache_file = os.path.join(
            cache_path,
            f'{split}_{rep}_cost_{cost}.parquet'
        ) 
        print(f'Caching transformed data to {cache_file}') 
        df.to_parquet(cache_file) 
        #df2 = pd.read_parquet(cache_file)


