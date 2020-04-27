"""Script to explore sepsis onsets"""
"""Author: Michael Moor"""

import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import glob
import os
import sys
import time
import pickle
from tqdm import tqdm
from IPython import embed
import seaborn as sns
import matplotlib.pyplot as plt


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def select_cols(df, time):
    selected = df[[time, 'SepsisLabel']]
    selected.set_index(time, inplace=True)
    return selected

#hard-coded path for prototyping:
def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    datasets = ['hirid', 'mimic3', 'eicu']
    times = ['datetime', 'charttime', 'observationoffset' ] #hirid, mimic3, eicu 
    filepath = "data/extracted/*.psv"
    data = defaultdict()
    #1. Iterate over datasets:
    #______________________
    for dataset, time in zip(datasets, times):
        base = os.path.join('datasets', dataset)
        path = os.path.join(base, filepath) 
    
        #2. Iterate over patient files:
        #------------------------------    
        print(f'Iterating over patient files of {dataset} dataset.. ')
        onsets = pd.DataFrame()
        case_counter = 0 
        for i, fname in enumerate(tqdm(glob.glob(path))):
            pat = pd.read_csv(fname, sep='|')
            if pat['SepsisLabel'].sum() == 0:
                continue #skip controls
            p = select_cols(pat, time)
            #find (first) onset: 
            onset_index = np.argmax(p)            
            onset_time = p.index[onset_index] 
            onsets = onsets.append(p[onset_index:onset_index+1])
            case_counter += 1
       
        onsets = onsets.reset_index()
        onsets = onsets.rename(columns = {time: 'time'})
        save_pickle(onsets, f'plots/case_onsets_{dataset}.pkl') 

if __name__ == "__main__":
    
    main()
