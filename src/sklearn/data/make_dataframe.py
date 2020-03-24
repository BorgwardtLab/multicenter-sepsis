import argparse
import os
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

from transformers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', 
        default='physionet2019',
        help='dataset to use: [physionet2019, ]')
    parser.add_argument('--data_dir', 
        default='../../../datasets',
        help='path pointing to the dataset directory')
    parser.add_argument('--out_dir',
        default='sklearn',
        help='relative path from <dataset>/data dir, where processed dump will be written to') 
    args = parser.parse_args()
    
    dataset = args.dataset
    base_dir = os.path.join(args.data_dir, dataset, 'data')

    #for Physionet2019, we use extracted folder:
    if dataset == 'physionet2019':
        data_dir = os.path.join(base_dir, 'extracted')
    else:
        data_dir = base_dir
 
    out_dir = os.path.join(base_dir, args.out_dir, 'processed')
    
    splits = ['train', 'validation'] # save 'test' for later
    
    for split in splits:
        # Run full pipe
        data_pipeline = Pipeline([
            ('create_dataframe', CreateDataframe(save=True, data_dir=out_dir, split=split )),
            #('input_count', AddRecordingCount()),
            ('normalization', Normalizer(data_dir=out_dir, split=split )),
            ('imputation', IndicatorImputation()),
            #('imputation', CarryForwardImputation()),
            #('derive_features', DerivedFeatures()),
            ('remove_nans', FillMissing())
        ])
        df = data_pipeline.fit_transform(data_dir)
        
        # Save
        save_pickle(df, os.path.join(out_dir, f'X_{split}.pkl'))

