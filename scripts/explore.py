import os
import pickle
import argparse
import sys
sys.path.append(os.getcwd())

from src.sklearn.data.utils import load_pickle
from IPython import embed

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to pickled df file to inspect', 
        default='datasets/physionet2019/data/sklearn/processed')
    args = parser.parse_args()
    df_path = os.path.join(args.path, 'X_features_train.pkl')
    y_path = os.path.join(args.path, 'y_train.pkl')

    df = load_pickle(df_path)
    y = load_pickle(y_path)

    embed()
