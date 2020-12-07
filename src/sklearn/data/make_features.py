import argparse
import os
import pandas as pd
import sys
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from time import time
from dask.distributed import Client
import dask.dataframe as dd


from .transformers import *
from .subsetters import ChallengeFeatureSubsetter 

dataset_class_mapping = {
    'physionet2019': 'Physionet2019Dataset',
    'mimic3': 'MIMIC3Dataset',
    'eicu': 'EICUDataset',
    'hirid': 'HiridDataset',
    'demo': 'DemoDataset',
    'aumc': 'AUMCDataset'
}

def ensure_single_index(df):
    df.reset_index(inplace=True)
    df.set_index(['id'], inplace=True)
    return df 

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', 
        default='physionet2019',
        help='dataset to use: [physionet2019, mimic3, eicu, hirid, demo]')
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
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    args = parser.parse_args()
    client = Client(n_workers=args.n_jobs, memory_limit='50GB', local_directory='/local0/tmp/dask')
    n_jobs = args.n_jobs
    overwrite = args.overwrite 
    dataset = args.dataset
    dataset_cls = dataset_class_mapping[dataset]

    base_dir = os.path.join(args.data_dir, dataset, 'data')

    data_dir = os.path.join(base_dir, 'extracted') #only used when creating df directly from psv
    out_dir = os.path.join(base_dir, args.out_dir, 'processed')
    splits = args.splits  
 
    #For verbosity, we outline preprocessing / filtering parameters here:
    cut_off = 24 #how many hours we include after a sepsis onset
    onset_bounds = (3, 168) #the included sepsis onset window (before too early, after too late)
     
    for split in splits:
        # Run full pipe
        print(f'Processing {split} split ..')
        
        #1. Fixed Data Pipeline: Dataloading and normalization (non-parallel)
        #--------------------------------------------------------------------
        # this step is not tunable, hence we cache it out in a pkl dump
        dump = os.path.join(out_dir, f'X_normalized_{split}.pkl')
        dump_for_deep = os.path.join(out_dir, f'X_filtered_{split}.pkl')
 
        if os.path.exists(dump) and not overwrite: 
            df = load_pickle(dump)
        else:
            print('Running (fixed) data pipeline and dumping it..')
            start = time()
            data_pipeline = Pipeline([
                ('create_dataframe', DataframeFromDataloader(save=True, dataset_cls=dataset_cls, data_dir=out_dir, 
                    split=split, drop_label=False)),
                ('drop_cases_with_late_or_early_onsets', PatientFiltration(save=True, data_dir=out_dir, 
                    split=split, onset_bounds=onset_bounds, n_jobs=n_jobs)),
                ('remove_time_after_sepsis_onset+window', CaseFiltrationAfterOnset(n_jobs=n_jobs, 
                    cut_off=cut_off, onset_bounds=onset_bounds, concat_output=True)),
                #('drop_labels', DropLabels(save=True, data_dir=out_dir, split=split)),
                ('derived_features', DerivedFeatures()),
                ('normalization', Normalizer(data_dir=out_dir, split=split)),
                ('categorical_one_hot_encoder', CategoricalOneHotEncoder())
            ])
            df = data_pipeline.fit_transform(None)
            print(f'.. finished. Took {time() - start} seconds.')
            # Save
            #### save_pickle(df, dump)
           
            # 1.B) Filtering for deep learning pipeline:
            #------------------------------------------ 
            # In addition, to prepare the data for the deep learning pipeline, 
            # consistently apply the same final invalid times filtration steps as in sklearn pipe
            # while skipping the manual feature engineering
            filter_for_deep_pipe =  Pipeline([
            ('filter_invalid_times', InvalidTimesFiltration()),
            ('drop_cols', DropColumns( save=False, ####True  
                data_dir=out_dir, split=split)), #save baselines here, as its not a dask df
             ])
            print('Running invalid times filtr. for deep pipeline..')
            df_deep = df.reset_index(level='time', drop=False) # invalid times filt. can't handle multi-index due to dask
            df_deep = filter_for_deep_pipe.fit_transform(df_deep) 
            #### save_pickle(df_deep, dump_for_deep)
            print('Done with invalid times filtr.')
    
        #2. Tunable Pipeline: Feature Extraction, further Preprocessing
        #---------------------------------------------------------------------------------
        # We need to sort the index by ourselves to ensure the time axis is
        # correctly ordered. Dask would not take this into account.
        df.sort_index(
            axis='index', level='id', inplace=True, sort_remaining=True)
        df.reset_index(level='time', drop=False, inplace=True)
        df = dd.from_pandas(df, npartitions=args.n_partitions, sort=True)
        print('Running (tunable) preprocessing pipeline and dumping it..')
        start = time()
        pipeline = Pipeline([
            ('lookback_features', LookbackFeatures(n_jobs=n_jobs)), ####concat_output=True)),
            ('measurement_counts', MeasurementCounter(n_jobs=n_jobs)),
            ('filter_invalid_times', InvalidTimesFiltration()),
            #drop and save baseline scores after filtering invalid (which ignored baselines)
            ('drop_cols', DropColumns(save=False))   # don't save here, as still delayed dask obj 
        ])
        df_deep2 = pipeline.fit_transform(df).compute()
        
        # Test how deep models perform with lookback features:
        # For sklearn pipe, we need proper multi index format once again 
        df_deep2.reset_index(inplace=True)
        df_deep2.set_index(['id', 'time'], inplace=True)
 
        sklearn_pipe =  Pipeline([
            ('imputation', IndicatorImputation(n_jobs=n_jobs, concat_output=True)),
            # wavelets require imputed data! 
            ('wavelet_features', WaveletFeatures(n_jobs=50, concat_output=True)), #n_jobs=n_jobs, concat_output=True 
            ('signatures', SignatureFeatures(n_jobs=10, concat_output=True)), #n_jobs=10
            ])
        df_sklearn = sklearn_pipe.fit_transform(df_deep2)

        print(f'.. finished. Took {time() - start} seconds.')
        
        #All models assume time as column and only id as index (multi-index would cause problem with dask models)
        df_deep2 = ensure_single_index(df_deep2)
        df_sklearn = ensure_single_index(df_sklearn) 

        # Save
        save_pickle(df_sklearn, os.path.join(out_dir, f'X_extended_features_{split}.pkl'))
        save_pickle(df_deep2, os.path.join(out_dir, f'X_extended_features_no_imp_{split}.pkl'))

    client.close()
    
    # Finally, we derive which features need to be dropped for physionet variable set:
    #### _ = ChallengeFeatureSubsetter(preprocessing=True)

if __name__ == '__main__':
    main()
