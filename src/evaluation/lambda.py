import numpy as np
import pandas as pd 
from joblib import Parallel, delayed

from IPython import embed
from src.sklearn.data.utils import load_pickle #save_pickle, index_check

def process_patient(df):
    pass

def compute_lambda(data, n_jobs=10):
    """
    Utility function to compute sample weight lambda for 
    prevalence correction of clinical utility score.
    """
    output = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=1)(
            delayed(process_patient)(data.loc[[i]]) for i in ids)


if __name__ == "__main__":

    
    features_path = 'datasets/physionet2019/data/sklearn/processed/X_features_train.pkl' 
    X = load_pickle(features_path)
    embed()
