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
    X_f_path = os.path.join(path, f'X_filtered_{split}.pkl')
    #df_n = load_pickle(normalized_path)
    X = load_pickle(features_path)
    X_f = load_pickle(X_f_path)

    stats = compute_stats(X_f) 
    print(stats)
    embed()
    #check_times(X_f)
    #embed()
