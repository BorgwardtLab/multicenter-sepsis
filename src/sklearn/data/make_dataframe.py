import argparse
import os
import pandas as pd
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
    parser.add_argument('--n_jobs', type=int,
        default=10,
        help='number of paralllel jobs') 
    args = parser.parse_args()
    
    dataset = args.dataset
    dataset_cls = dataset_class_mapping[dataset]

    base_dir = os.path.join(args.data_dir, dataset, 'data')

    data_dir = os.path.join(base_dir, 'extracted') #only used when creating df directly from psv
    out_dir = os.path.join(base_dir, args.out_dir, 'processed')
    splits = ['train', 'validation'] # save 'test' for later
     
    for split in splits:
        # Run full pipe
        print(f'Processing {split} split ..')
        start = time() 
        #1. Dataloading and normalization (non-parallel)
        data_pipeline = Pipeline([
            ('create_dataframe', DataframeFromDataloader(save=True, dataset_cls=dataset_cls, data_dir=out_dir, split=split)),
            #('create_dataframe', CreateDataframe(save=True, data_dir=out_dir, split=split)),
            ('normalization', Normalizer(data_dir=out_dir, split=split)),
            ('lookback_features', LookbackFeatures(n_jobs=args.n_jobs)),
            #('imputation', IndicatorImputation()),
            ('imputation', CarryForwardImputation()),
            #('derive_features', DerivedFeatures()),
            ('remove_nans', FillMissing())
        ])
        df = data_pipeline.fit_transform(data_dir) #the dir argument is inactive, except when using CreateDataFrame 
        #(using the psv files as source, instead of iterable dataset class
        print(f'.. finished. Took {time() - start} seconds.')
 
        # Save
        save_pickle(df, os.path.join(out_dir, f'X_{split}.pkl'))

if __name__ == '__main__':
    main()


