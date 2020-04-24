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
    raw_data_path = os.path.join(path, f'raw_data_{split}.pkl')
    normalized_path = os.path.join(path, f'X_normalized_{split}.pkl')
    features_path = os.path.join(path, f'X_features_{split}.pkl')
    y_path = os.path.join(path, f'y_{split}.pkl')
    raw_y_path =  os.path.join(path, f'raw_y_{split}.pkl')
    df = load_pickle(raw_data_path)
    df_n = load_pickle(normalized_path)
    X = load_pickle(features_path)
    y = load_pickle(y_path)
    raw_y = load_pickle(raw_y_path)
    embed()
