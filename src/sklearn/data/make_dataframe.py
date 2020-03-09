import argparse
import os
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from transformers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', 
        default='../../datasets/physionet2019/data',
        help='path pointing to /data directory')
    parser.add_argument('--out_dir',
        default='sklearn',
        help='relative path from /data dir, where processed dump will be written to') 
    args = parser.parse_args()

    base_dir = args.data_dir

    #for Physionet2019, we use extracted folder:
    if 'physionet2019' in base_dir:
        data_dir = os.path.join(base_dir, 'extracted')
    else:
        data_dir = base_dir 
    out_dir = os.path.join(base_dir, args.out_dir)
    

    # Run full pipe
    data_pipeline = Pipeline([
        ('create_dataframe', CreateDataframe(save=True, data_dir=out_dir))
    ])
    df = data_pipeline.fit_transform(data_dir)
    
    # Save
    save_pickle(df, out_dir + '/df.pickle')

