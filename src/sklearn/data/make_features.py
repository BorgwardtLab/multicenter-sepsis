import argparse
import os
import pandas as pd
import sys
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from time import time
from dask.distributed import Client
import dask.dataframe as dd
import json

from .transformers import *
from .subsetters import ChallengeFeatureSubsetter 
from src.variables.mapping import VariableMapping


def ensure_single_index(df):
    df.reset_index(inplace=True)
    df.set_index(['id'], inplace=True)
    return df 

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', 
        default='mimic',
        help='dataset to use: [physionet2019, mimic, eicu, hirid, aumc, demo]')
    parser.add_argument('--data_dir', 
        default='datasets',
        help='path pointing to the dataset directory')
    parser.add_argument('--out_dir',
        default='sklearn',
        help='relative path from <dataset>/data dir, where processed dump will be written to') 
    parser.add_argument('--n_jobs', type=int,
        default=10,
        help='number of paralllel jobs to process the full df in chunks')
    parser.add_argument('--overwrite', action='store_true',
        default=False,
        help='compute all preprocessing steps and overwrite existing intermediary files')
    parser.add_argument('--n_partitions', type=int,
        default=50,
        help='number of df partitions in dask')
    parser.add_argument('--n_chunks', type=int,
        default=1,
        help='number of df chunks for wavelet and signature extraction (relevant for eICU)')
    parser.add_argument('--var_config_path', 
        default='config/variables.json',
        help='path to config info (variable names)')
    parser.add_argument('--split_config_path', 
        default='config/master_splits.json',
        help='path split info')

    args = parser.parse_args()
    client = Client(n_workers=args.n_jobs, memory_limit='50GB', local_directory='/local0/tmp/dask')
    n_jobs = args.n_jobs
    overwrite = args.overwrite 
    dataset = args.dataset

    base_dir = os.path.join(args.data_dir, dataset, 'data')

    data_dir = os.path.join(base_dir, 'extracted') #only used when creating df directly from psv
    out_dir = os.path.join(base_dir, args.out_dir, 'processed')
    var_config_path = args.var_config_path
    split_config_path = args.split_config_path
 
    # we initialize a dynamic variable mapping object which can be passed through the preprocessing transforms 
    vm = VariableMapping(
        variable_file = var_config_path,
        input_col = 'concept', # we stick to the ricu concept
        output_col = 'name' #currently used variable names, this can be changed
    )
    with open(split_config_path, 'r') as f:
        split_info = json.load(f) 
    split_info = split_info[dataset] 
     
    data_path = 'datasets/downloads/mimic_demo_int.parquet'
    
    #1. Fixed Data Pipeline: Dataloading, Derived Features (non-parallel)
    #--------------------------------------------------------------------
    # this step is not tunable, hence we cache it out in a pkl dump
    print('Running (fixed) data pipeline and dumping it..')
    start = time()
    data_pipeline = Pipeline([
        ('create_dataframe', DataframeFromParquet(data_path, vm)),  
        ('derived_features', DerivedFeatures(vm, suffix='locf')),
    ])
    df = data_pipeline.fit_transform(None)
    
    print(f'.. finished. Took {time() - start} seconds.')
   
    #2. Feature Extraction, further Preprocessing
    #---------------------------------------------------------------------------------
    # We need to sort the index by ourselves to ensure the time axis is
    # correctly ordered. Dask would not take this into account.
    df.sort_index(
        axis='index', level=vm('id'), inplace=True, sort_remaining=True)
    df.reset_index(level=vm('time'), drop=False, inplace=True)
    df = dd.from_pandas(df, npartitions=args.n_partitions, sort=True)
    print('Running (tunable) preprocessing pipeline and dumping it..')
    start = time()
    dask_pipeline = Pipeline([
        ('lookback_features', LookbackFeatures(vm=vm,  
            suffices=['_raw', '_derived'])), ####concat_output=True)),
        ('measurement_counts', MeasurementCounter(vm=vm, suffix='_raw')),
    #    ('filter_invalid_times', InvalidTimesFiltration()),
    ])
    df = dask_pipeline.fit_transform(df).compute()

    # For sklearn pipe, we need proper multi index format once again 
    df.reset_index(inplace=True)
    df.set_index([vm('id'), vm('time')], inplace=True)

    # We chunk for the next memory-costly part (which could not easily be implemented in dask)
    ids = np.unique(df.index.get_level_values(vm('id')))
    id_splits = np.array_split(ids, args.n_chunks)
    df_splits = {}
    for i, id_split in enumerate(id_splits):
        df_splits[i] = df.loc[id_split]  # list comp. didn't find df in scope
    # clear large df from memory: 
    del df 
    #TODO: add invalid times filtration! 
    pandas_pipeline =  Pipeline([
        ('imputation', IndicatorImputation(n_jobs=n_jobs, suffix='_raw', concat_output=True)),
        ('feature_normalizer', Normalizer(split_info, split='dev', 
            suffix=['_locf', '_derived'])),
        # wavelets require imputed data --> we pad 0s in locf nans on the fly 
        ('wavelet_features', WaveletFeatures(n_jobs=n_jobs, suffix='_locf', 
            concat_output=True)), #n_jobs=5, concat_output=True 
        ('signatures', SignatureFeatures(n_jobs=n_jobs, 
            suffices=['_locf', '_derived'], concat_output=True)), #n_jobs=2
        ])
    out_df_splits = []
    keys = list(df_splits.keys()) # dict otherwise doesnt like popping keys
    for key in keys:
        out_df = pandas_pipeline.fit_transform(df_splits[key])
        #free mem of current input df:
        df_splits.pop(key)
        out_df_splits.append(out_df) 
    df = pd.concat(out_df_splits)

    print(f'.. finished. Took {time() - start} seconds.')
    
    from IPython import embed; embed(); sys.exit() 
    #All models assume time as column and only id as index (multi-index would cause problem with dask models)
    ##df_deep2 = ensure_single_index(df_deep2)
    df = ensure_single_index(df) 

    # Save
    #save_pickle(df_sklearn, os.path.join(out_dir, f'X_extended_features_{split}.pkl'))
    #save_pickle(df_deep2, os.path.join(out_dir, f'X_extended_features_no_imp_{split}.pkl'))

    client.close()
    
    # Finally, we derive which features need to be dropped for physionet variable set:
    #### _ = ChallengeFeatureSubsetter(preprocessing=True)

if __name__ == '__main__':
    main()
