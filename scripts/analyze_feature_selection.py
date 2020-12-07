import pandas as pd
from IPython import embed
import pickle
import joblib
import argparse
import numpy as np

def align_scores_with_features(scores, fs):
    for score, feature_set in zip(scores, fs[0]): #for length first fold of fs is enough
        print(f' Score: {score:.4f}, #Features: {len(feature_set)}') 

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='results/feature_selection/feature_selection_demo_lgbm/rfe.pkl') 
#aumc: results/feature_selection/signatures_wavelets_counts_GAIN/feature_selection_aumc_lgbm/rfe.pkl
parser.add_argument('--data_path', default='datasets/demo/data/sklearn/processed/X_extended_features_validation.pkl') #aumc

args = parser.parse_args()

with open(args.data_path, 'rb') as f:
    df = pickle.load(f)

with open(args.input_path, 'rb') as f:
    rfe = joblib.load(f)

align_scores_with_features(rfe.grid_scores_, rfe.grid_feature_sets_)

embed()
