import os
import pickle
import argparse
import sys
sys.path.append(os.getcwd())
from src.sklearn.data.utils import load_pickle
from IPython import embed
import json
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

def check_times(df):
    patients = df.index.unique()
    for pat_id in patients:
        patient_data = df.loc[pat_id]
        try:
            time = patient_data['time'].values
        except:
            print('Problem occured!')
            embed()
        print(f'Patient {pat_id} has times {time}')

def compute_stats(df, results, dataset, split):
    """
    Computes case and case timepoint frequencies
    """
    #Time point prevalence of sepsis:
    tp_freq = df['sep3'].sum()/df['sep3'].shape    
    results['tp_prevalence'].append(tp_freq[0])
    #Case-level prevalence:
    case_ids = df[df['sep3'] == 1].index.unique() 
    total_ids = df.index.unique()
    freq = len(case_ids) / len(total_ids)
    results['case_prevalence'].append(freq)
    results['total_stays'].append(len(total_ids))
    results['total_cases'].append(len(case_ids))
    results['dataset'].append(dataset)
    results['split'].append(split)
    return results

def check_nans(df, name):
    n_nans = df.isnull().sum().sum()
    if n_nans > 0:
        print(f'{name} has {n_nans}')
        print(df.isnull().sum())
    else:
        print(f'No nans found in {name}') 

def print_shape(df, name):
    print(f'Shape of {name}: {df.shape}') 

def check_df(df, name):
    check_nans(df, name)
    print_shape(df, name)
 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path from dataset to pickled df file to inspect', 
        default='data/sklearn/processed')
    parser.add_argument('--dataset', type=str, help='dataset to use, when setting `all`, all datasets are iterated over', 
        default='physionet2019')
    parser.add_argument('--split', type=str, help='split to use when setting `all`, all datasets are iterated over', 
        default='validation')

    args = parser.parse_args()
    split = args.split 
    dataset = args.dataset

    #check if looping over all splits and datasets or just single provided one
    if split == 'all':
        splits = ['train','validation', 'test']
    else:
        splits = [split]

    if dataset == 'all':
        datasets = ['physionet2019', 'mimic3', 'eicu', 'hirid', 'aumc']
    else:
        datasets = [dataset]

    results = {}
    #initialize keys:
    results['tp_prevalence'] = [] 
    results['case_prevalence'] = [] 
    results['total_stays'] = []
    results['total_cases'] = []
    results['dataset'] = []
    results['split'] = []

    lengths = 0
    for dataset in datasets: 
        
        for split in splits:
            
            path = os.path.join('datasets', dataset, args.path)
            features_path = os.path.join(path, f'X_features_{split}.pkl')
            filtered_path = os.path.join(path, f'X_filtered_{split}.pkl')

            X_ni_path = os.path.join(path, f'X_features_no_imp_{split}.pkl')
            baseline_path = os.path.join(path, f'baselines_{split}.pkl')
         
            #X = load_pickle(features_path)
            X_f = load_pickle(filtered_path)  
            #b = load_pickle(baseline_path)
        
            lengths += len(X_f) 
            #dfs = [X, Xf]
            #names = ['X_features', 'X_features_no_imp']
            #for df, name in zip(dfs, names):
            #    check_df(df, name)
         
            results = compute_stats(X_f, results, dataset, split) 
            #check_times(X_f)
    
    df = pd.DataFrame(results)
    df = df.sort_values(by=['tp_prevalence'])
    fig, ax = plt.subplots(1,2, figsize=(12,5), sharey='all')
    sns.barplot(ax=ax[0], x='dataset', y='tp_prevalence', data=df)  
    sns.barplot(ax=ax[1], x='dataset', y='case_prevalence', data=df)
    fig.suptitle('Prevalence on time point and case level', fontsize=16)
    plt.savefig( os.path.join('results', 'pos_weight_analysis', 'prevalences.png'), dpi=300)
 
    out_path = os.path.join('results', 'pos_weight_analysis', 'dataset_stats.csv')
    df.to_csv(out_path)
