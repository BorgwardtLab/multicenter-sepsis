import os
import pickle
import argparse
import sys
sys.path.append(os.getcwd())
from src.sklearn.data.utils import load_pickle
from IPython import embed

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

def compute_stats(df):
    """
    Computes case and case timepoint frequencies
    """
    results = []
    #Time point prevalence of sepsis:
    tp_freq = X_f['SepsisLabel'].sum()/X_f['SepsisLabel'].shape    
    results.append(tp_freq[0])
    #Case-level prevalence:
    case_ids = X_f[X_f['SepsisLabel'] == 1].index.unique() 
    total_ids = X_f.index.unique()
    freq = len(case_ids) / len(total_ids)
    results.append(freq)
    results.append(len(total_ids))
    results.append(len(case_ids))
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
    parser.add_argument('--dataset', type=str, help='dataset to use', 
        default='physionet2019')
    parser.add_argument('--split', type=str, help='split to use', 
        default='validation')

    args = parser.parse_args()
    split = args.split 
    path = os.path.join('datasets', args.dataset, args.path)
    #normalized_path = os.path.join(path, f'X_normalized_{split}.pkl')
    features_path = os.path.join(path, f'X_features_{split}.pkl')
    X_f_path = os.path.join(path, f'X_features_no_imp_{split}.pkl')
    baseline_path = os.path.join(path, f'baselines_{split}.pkl')
 
    #df_n = load_pickle(normalized_path)
    X = load_pickle(features_path)
    Xf = load_pickle(X_f_path)
    b = load_pickle(baseline_path)
 
    #dfs = [X, Xf]
    #names = ['X_features', 'X_features_no_imp']
    #for df, name in zip(dfs, names):
    #    check_df(df, name)
 
    #stats = compute_stats(X_f) 
    #print(stats)
    embed()
    #check_times(X_f)
    #embed()
