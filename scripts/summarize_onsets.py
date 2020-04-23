"""Script to summarize onsets"""
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
 
def load_pickle(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d

def cumulative_counts(df):
    def above(series, thres):
        return len(series[ series > thres])
    def below_equal(series, thres):
        return len(series[ series <= thres])
    counts = {'time': [], 'onsets_after': [], 'onsets_before_equal': []}
    times = df['time']
    minimum = times.min()
    maximum = times.max()
    thresholds = np.arange(minimum, maximum+1) 
    for thres in thresholds:
        counts['time'].append(thres)
        counts['onsets_after'].append( above(times, thres) )
        counts['onsets_before_equal'].append(below_equal(times,thres))
    return pd.DataFrame.from_dict(counts)

#hard-coded path for prototyping:
def main():
    datasets = ['hirid', 'mimic3', 'eicu']
    data = defaultdict()
    #1. Iterate over datasets:
    #______________________
    for dataset in datasets:
        path = os.path.join('plots', f'case_onsets_{dataset}.pkl')
        if not os.path.exists(path):
            print(f'{dataset} onsets not yet extracted. Skipping..')
            continue
        else:
            data = load_pickle(path)
            assert data['SepsisLabel'].sum() == len(data) #SepsisLabel should always be 1
            counts = cumulative_counts(data) 
            print(f'For {dataset} Dataset, counts of onsets after given ICU time:')
            onset_out = counts[counts['time'].isin(np.arange(-1,20))] 
            onset_out.reset_index(inplace=True, drop=True)
            print(onset_out)
            ###plot label as bars:
            #plt.figure(figsize=(15,15))
            #sns.distplot(data['time'])  
            ##sns.distplot(case_data['time'], hist_kws={"weights": case_data['SepsisLabel']}, norm_hist=False, kde=False,
            ##bins=500)
            ##sns.jointplot("time", "SepsisLabel", data=case_data) #, kind="kde")
            ##sns.barplot(x='time', y='SepsisLabel', data=case_data)
            #plt.savefig(f'plots/onset_histogram_{dataset}.png')

if __name__ == "__main__":
    
    main()
