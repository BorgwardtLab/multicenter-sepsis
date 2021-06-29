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
from src.sklearn.loading_utils import SplitInfo, ParquetLoader
from src.variables.feature_groups import ColumnFilter


VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)
VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


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
            print(f'Custom feature set {feature_set} not among cached feature sets, trying to compute it on the fly!')
            if feature_set == 'locf':
                cf = ColumnFilter()
                cols = cf.feature_set(name='small', groups=False) 
                cols += cf.feature_set(name='locf', groups=False)
                cols = list(np.unique(cols))
                print('For custom feature set, variable_set (physionet or not) is ignored!')
            else:
                raise NotImplementedError(f'feature set {feature_set} not implemented with caching..')
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


