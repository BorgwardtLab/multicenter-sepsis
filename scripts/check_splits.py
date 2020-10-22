import os
import pickle
import argparse
import sys
sys.path.append(os.getcwd())

from src.sklearn.data.utils import load_pickle
from IPython import embed
import numpy as np
 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path from dataset to pickled df file to inspect', 
        default='data/sklearn/processed')
    parser.add_argument('--dataset', type=str, help='dataset to use', 
        default='physionet2019')
    
    args = parser.parse_args()
    datasets = ['demo', 'physionet2019', 'mimic3', 'eicu', 'hirid']
    splits = ['train', 'validation', 'test'] 
    for dataset in datasets: 
        for split in splits: 
            path = os.path.join('datasets', dataset, args.path)
            split_file = os.path.join('datasets', dataset, 'data', 'split_info.pkl')
            features_path = os.path.join(path, f'X_features_{split}.pkl')
            X = load_pickle(features_path)
            d = load_pickle(split_file)
            patients = d['split_0'][split]
            patients_after = X.index.unique()
            n_after = len(patients_after)
            n_before = len(patients)
            
            #Determine cases numbers using split info:
            id_to_label = {pid: label for pid, label in zip(d['pat_ids'], d['labels']) }
            prev_before = np.sum([ id_to_label[pid] for pid in patients]) / n_before
            prev_after = np.sum([ id_to_label[pid] for pid in patients_after]) / n_before
            print(f'{dataset}, {split} split, #patients / prevalence before prepro: {n_before} / {prev_before},  after prepro: {n_after} / {prev_after}')  
            
