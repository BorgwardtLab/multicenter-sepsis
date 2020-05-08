import os
import pickle
import argparse
import sys
sys.path.append(os.getcwd())

from src.sklearn.data.utils import load_pickle
from IPython import embed

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

    embed()
