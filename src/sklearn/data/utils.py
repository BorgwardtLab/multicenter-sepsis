import os
import pickle
import json
from collections import defaultdict

def save_pickle(obj, filename, protocol=4):
    """ Basic pickle dumping """
    _create_folder_if_not_exist(filename)
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=protocol)

def load_pickle(filename):
    """ Basic pickle loading function """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj

def _create_folder_if_not_exist(filename):
    """ Makes a folder if the path does not already exist """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

def load_json(filename):
    with open(filename) as file:
        obj = json.load(file)
    return obj


def load_data(path='datasets/physionet2019/data/sklearn/processed'):
    """ 
    Load preprocessed Data in sklearn format from pickle
    """ 
    data = defaultdict()
    file_renaming = { 
                'X_features_train': 'X_train',
                'y_train': 'y_train', 
                'X_features_validation': 'X_validation',
                'y_validation': 'y_validation'
    } 
    
    for filename in file_renaming.keys():
        filepath = os.path.join(path, filename + '.pkl')
        data[file_renaming[filename]] = load_pickle(filepath)
    return data

def to_tsfresh_form(df, labels):
    """ Puts the data into a form where tsfresh extractors can be used. """
    tsfresh_frame = df.reset_index()
    labels = labels.groupby('id').apply(lambda x: x.iloc[0]).astype(int)
    return tsfresh_frame, labels
 
 
