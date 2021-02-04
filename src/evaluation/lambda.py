import numpy as np
import pandas as pd 
from joblib import Parallel, delayed

from IPython import embed
from src.sklearn.data.utils import load_pickle #save_pickle, index_check

class LambdaCalculator:
    """
    Class for computing sample weight lambda for 
    prevalence correction of clinical utility score.

    """
    def __init__(self, n_jobs=10, u_fp=-0.05, early_window=12, label='sep3'):
        """
        Inputs: 
            - n_jobs: number of jobs for parallelism
            - u_fp: U_{FP} value as used in util score
            - early_window: duration from t_early to t_sepsis 
                in util score
        """
        self.n_jobs = n_jobs
        self.u_fp = u_fp
        self.early_window = early_window
        self.label = label 

    def _process_patient(self, df):
        t_minus = 0
        t_plus = 0
        y = df[self.label]

        # we assume that there are no remaining NaNs! assert this
        n_nans = df.isnull().sum().sum()
        assert n_nans == 0

        is_septic = 0
        if np.any(y): # if septic case
            is_septic = 1
            onset = np.argmax(y)
            t_plus = max(0, onset - self.early_window)
        else: # if control
            t_minus = len(y)
        return t_minus, t_plus, is_septic

    def __call__(self, data):
        """
        Inputs:
            - data: pandas df of dataset
        """
        u_fp = self.u_fp
        ids = data.index.get_level_values(0) #works both on single and multi index 
        # level, assuming that ids are first index 
        t_minus = 0
        t_plus = 0
        septic = 0
        output = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=1)(
                delayed(self._process_patient)(data.loc[[i]]) for i in ids)
        for (t_m, t_p, s) in output:
            t_minus += t_m
            t_plus += t_p
            septic += s
            #sanity check: at most one can be non-zero:
            assert t_m * t_p == 0
        lam = (u_fp * t_minus) / (33 * septic - u_fp * t_plus) 
        
        self.lam = lam
        self.t_minus = t_minus
        self.t_plus = t_plus
        self.septic = septic
        self.prev = septic / len(output)

        return lam

if __name__ == "__main__":

    features_path = 'datasets/physionet2019/data/sklearn/processed/X_features_train.pkl' 
    X = load_pickle(features_path)
    
    calc = LambdaCalculator(n_jobs=30)
    lam = calc(X)
    embed()
