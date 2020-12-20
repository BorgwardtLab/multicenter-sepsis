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
    results = {}
    #Time point prevalence of sepsis:
    tp_freq = df['sep3'].sum()/df['sep3'].shape    
    results['tp_prevalence'] = tp_freq[0]
    #Case-level prevalence:
    case_ids = df[df['sep3'] == 1].index.unique() 
    total_ids = df.index.unique()
    freq = len(case_ids) / len(total_ids)
    results['case_prevalence'] = freq
    results['total_stays'] = len(total_ids)
    results['total_cases'] = len(case_ids)
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
        datasets = ['demo', 'physionet2019', 'mimic3', 'eicu', 'hirid', 'aumc']
    else:
        datasets = [dataset]

    results = {}
    lengths = 0
    for dataset in datasets: 
        results[dataset] = {}
        
        for split in splits:
            
            path = os.path.join('datasets', dataset, args.path)
            #normalized_path = os.path.join(path, f'X_normalized_{split}.pkl')
            features_path = os.path.join(path, f'X_features_{split}.pkl')
            filtered_path = os.path.join(path, f'X_filtered_{split}.pkl')

            X_ni_path = os.path.join(path, f'X_features_no_imp_{split}.pkl')
            baseline_path = os.path.join(path, f'baselines_{split}.pkl')
         
            #df_n = load_pickle(normalized_path)
            X = load_pickle(features_path)
            X_ni = load_pickle(X_ni_path)
            X_f = load_pickle(filtered_path)  
            b = load_pickle(baseline_path)
        
            lengths += len(X_f) 
            #dfs = [X, Xf]
            #names = ['X_features', 'X_features_no_imp']
            #for df, name in zip(dfs, names):
            #    check_df(df, name)
         
            results[dataset][split] = compute_stats(X) 
            #check_times(X_f)

    overall_total = 0
    overall_cases = 0 
    for dataset in results.keys():
        total = 0
        cases = 0 
        for split in results[dataset].keys():
            total += results[dataset][split]['total_stays']
            cases += results[dataset][split]['total_cases']
        overall_total += total
        overall_cases += cases
        print(dataset, total, cases)
    print(overall_total, overall_cases)
    print(lengths/(24*365), 'years')
    embed()

