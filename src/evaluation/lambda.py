import numpy as np
import pandas as pd 
from joblib import Parallel, delayed

from IPython import embed
from src.variables.mapping import VariableMapping
from src.sklearn.data.utils import load_pickle #save_pickle, index_check
from .physionet2019_score import compute_prediction_utility 
from .sklearn_utils import make_consecutive

class LambdaCalculator:
    """
    Class for computing sample weight lambda for 
    prevalence correction of clinical utility score.

    """
    def __init__(self, n_jobs=10, n_chunks=50, u_fp=-0.05, early_window=12, vm=None):
        """
        Inputs: 
            - n_jobs: number of jobs for parallelism
            - u_fp: U_{FP} value as used in util score
            - early_window: duration from t_early to t_sepsis 
                in util score
            - vm: Variable Mapping obj
        """
        self.n_jobs = n_jobs
        self.n_chunks = n_chunks
        self.u_fp = u_fp
        self.early_window = early_window
        self.vm = vm 

    def _process_chunk_of_patients(self, data):
        #initialize outputs:
        t_minus_tot = 0
        t_plus_tot = 0
        g_tot = 0
        h_tot = 0
        n_septic = 0 

        ids = data.index.get_level_values(0).unique()
        for i in ids:
            df = data.loc[[i]] 
            df = df.reset_index().set_index(self.vm('time')) 
 
            y = df[self.vm('label')]

            # we assume that there are no remaining NaNs! assert this
            n_nans = df.isnull().sum().sum()
            assert n_nans == 0
            
            # ensure that time series is consecutive:
            y, _ = make_consecutive(y,y)
 
            is_septic = 0
            t_plus = 0
            t_minus = 0
            g = 0
            h = 0 
            if np.any(y): # if septic case
                is_septic = 1
                onset = np.argmax(y)
                t_plus = max(0, onset - self.early_window)
                g = compute_prediction_utility(
                        y, np.ones(len(y)), u_fp=self.u_fp, return_all_scores=True)
                g = np.maximum(g,0) #here we only need the non-negative utilities
                g = g.sum()
                h = compute_prediction_utility(
                        y, np.zeros(len(y)), u_fp=self.u_fp, return_all_scores=True)
                h = h.sum()
            else: # if control
                t_minus = len(y)
            #per patient only one can be non-zero:
            assert t_plus * t_minus == 0
            t_minus_tot += t_minus
            t_plus_tot += t_plus
            g_tot += g
            h_tot += h
            n_septic += is_septic

        return t_minus_tot, t_plus_tot, n_septic, g_tot, h_tot
 
    def __call__(self, data):
        """
        Inputs:
            - data: pandas df of dataset
        """
        def collate_ids(df, ids):
            return pd.concat([ df.loc[[i]] for i in ids])
        
        u_fp = self.u_fp
        ids = data.index.get_level_values(0).unique() #works both on single and multi index
        id_chunks = np.array_split(ids, self.n_chunks)
        # level, assuming that ids are first index 
        t_minus = 0
        t_plus = 0
        septic = 0
        g = 0
        h = 0 
        output = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self._process_chunk_of_patients)(collate_ids(data, chunk)) for chunk in id_chunks)
        for (t_m, t_p, s, g_, h_) in output:
            t_minus += t_m
            t_plus += t_p
            septic += s
            g += g_
            h += h_

        lam = (-u_fp * t_minus) / ( 2 * g - 2 * h + u_fp * t_plus) 
        
        self.lam = lam
        self.t_minus = t_minus
        self.t_plus = t_plus
        self.septic = septic
        self.prev = septic / len(output)

        return lam


class SimplifiedLambdaCalculator:
    """
    Class for computing simplified sample weight lambda for 
    prevalence correction of clinical utility score.
    Simplification: we assume that the critical window in sepsis patients
    is fully observed.

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
        ids = data.index.get_level_values(0).unique() #works both on single and multi index
        # level, assuming that ids are first index 
        t_minus = 0
        t_plus = 0
        septic = 0
        output = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self._process_patient)(data.loc[[i]]) for i in ids)
        for (t_m, t_p, s) in output:
            t_minus += t_m
            t_plus += t_p
            septic += s
            #sanity check: at most one can be non-zero:
            assert t_m * t_p == 0
        lam = (-u_fp * t_minus) / (33 * septic + u_fp * t_plus) 
        
        self.lam = lam
        self.t_minus = t_minus
        self.t_plus = t_plus
        self.septic = septic
        self.prev = septic / len(output)

        return lam

if __name__ == "__main__":
    result = {}
    datasets = ['demo']
    path = 'datasets/{}/data/parquet/features.parquet'
    vm = VariableMapping()
    for dataset in datasets:
        X = pd.read_parquet(path.format(dataset), columns=['sep3', 'stay_time'])
        calc = LambdaCalculator(n_jobs=1, vm=vm)
        lam = calc(X)
        result[dataset] = lam
    print(result)
    embed()

    
