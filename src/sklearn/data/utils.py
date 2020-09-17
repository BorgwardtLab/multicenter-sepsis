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

def index_check(full_data):
    #Sanity check that time index is properly set: FIXME this has to be ensured at the end of preprocessing!
    index_cols = ['id', 'time']
    if any([col in full_data.columns for col in index_cols]):
        print('Formatting multi-index propery!..')
        full_data.reset_index(inplace=True)
        full_data.set_index(['id', 'time'], inplace=True) 
    return full_data

def load_data(path='datasets/physionet2019/data/sklearn/processed', label='sep3', index='multi'):
    """ 
    Load preprocessed Data in sklearn format from pickle, depending on index type reformat properly.
    """
    splits = ['train', 'validation']
    drop_col = 'Gender_Other' #only 5 patients in eicu have this, dropping this col as its still encoded in male female both zero. 
    data = defaultdict()
    files = ['X_features_' + split for split in splits]
    
    for split, filename in zip(splits, files):
        filepath = os.path.join(path, filename + '.pkl')
        full_data = load_pickle(filepath)
        if index == 'multi':
            full_data = index_check(full_data)
            print('Multi-index check finished')
        elif index == 'single':
            print('Single-index is used')
        else:
            raise NotImplementedError(f'{index} not among valid index types [multi, single]')        
        y = full_data[label]
        X = full_data.drop(label, axis=1)
        if drop_col in X.columns:
            X = X.drop(columns=drop_col, axis=1)
            print(f'Shape after dropping {drop_col}: {X.shape}')    
        data[f'X_{split}'] = X
        data[f'y_{split}'] = y
    return data

def to_tsfresh_form(df, labels):
    """ Puts the data into a form where tsfresh extractors can be used. """
    tsfresh_frame = df.reset_index()
    labels = labels.groupby('id').apply(lambda x: x.iloc[0]).astype(int)
    return tsfresh_frame, labels
 
 
