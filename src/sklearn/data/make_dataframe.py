import argparse
import os
import pandas as pd
import sys
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from time import time

from transformers import *

dataset_class_mapping = {
    'physionet2019': 'Physionet2019Dataset',
    'mimic3': 'MIMIC3Dataset',
    'test': None
}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', 
        default='physionet2019',
        help='dataset to use: [physionet2019, ]')
    parser.add_argument('--data_dir', 
        default='datasets',
        help='path pointing to the dataset directory')
    parser.add_argument('--out_dir',
        default='sklearn',
        help='relative path from <dataset>/data dir, where processed dump will be written to') 
    parser.add_argument('--n_outer_jobs', type=int,
        default=10,
        help='number of paralllel jobs to process the full df in chunks')
    parser.add_argument('--n_inner_jobs', type=int,
        default=1,
        help='number of paralllel jobs to compute different stats on one df chunk')
    parser.add_argument('--overwrite', action='store_true',
        default=False,
        help='compute all preprocessing steps and overwrite existing intermediary files') 
    args = parser.parse_args()
    
    overwrite = args.overwrite 
    dataset = args.dataset
    dataset_cls = dataset_class_mapping[dataset]

    base_dir = os.path.join(args.data_dir, dataset, 'data')

    data_dir = os.path.join(base_dir, 'extracted') #only used when creating df directly from psv
    out_dir = os.path.join(base_dir, args.out_dir, 'processed')
    splits = ['train', 'validation'] # save 'test' for later 
     
    for split in splits:
        # Run full pipe
        print(f'Processing {split} split ..')
        
        #1. Fixed Data Pipeline: Dataloading and normalization (non-parallel)
        #--------------------------------------------------------------------
        # this step is not tunable, hence we cache it out in a pkl dump
        dump = os.path.join(out_dir, f'X_normalized_{split}.pkl')
        if os.path.exists(dump) and not overwrite: 
            df = load_pickle(dump)
        else:
            print('Running (fixed) data pipeline and dumping it..')
            start = time() 
            data_pipeline = Pipeline([
                ('create_dataframe', DataframeFromDataloader(save=True, dataset_cls=dataset_cls, data_dir=out_dir, split=split)),
                ('derived_features', DerivedFeatures()),
                ('normalization', Normalizer(data_dir=out_dir, split=split))
            ])
            df = data_pipeline.fit_transform(None) 
            print(f'.. finished. Took {time() - start} seconds.')
            # Save
            save_pickle(df, dump)

        #2. Tunable Pipeline: Feature Extraction, further Preprocessing and Classification
        #---------------------------------------------------------------------------------
        print('Running (tunable) preprocessing pipeline and dumping it..')
        start = time()
        pipeline = Pipeline([
            ('lookback_features', LookbackFeatures(n_outer_jobs=args.n_outer_jobs, n_inner_jobs=args.n_inner_jobs)),
            ('imputation', CarryForwardImputation()),
            ('remove_nans', FillMissing())
        ])
        df = pipeline.fit_transform(df)  
        print(f'.. finished. Took {time() - start} seconds.')
        
        # Save
        save_pickle(df, os.path.join(out_dir, f'X_features_{split}.pkl'))

if __name__ == '__main__':
    main()


