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

    args = parser.parse_args()
    
    path = os.path.join('datasets', args.dataset, args.path)
    df_path = os.path.join(path, 'X_normalized_train.pkl')
    y_path = os.path.join(path, 'y_train.pkl')

    df = load_pickle(df_path)
    y = load_pickle(y_path)

    embed()
