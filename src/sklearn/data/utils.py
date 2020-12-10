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

def handle_index(data, index):
    if index == 'multi':
        data = index_check(data)
        print('Multi-index check finished')
    elif index == 'single':
        print('Single-index is used')
    else:
        raise NotImplementedError(f'{index} not among valid index types [multi, single]')
    return data

def load_data(
    path='datasets/physionet2019/data/sklearn/processed',
    label='sep3',
    splits = ['train', 'validation', 'test'],
    index='multi', 
    extended_features=False):
    """
    Load preprocessed Data in sklearn format from pickle, depending on index type reformat properly.
    """
    drop_col = 'Gender_Other' #only 5 patients in eicu have this, dropping this col as its still encoded in male female both zero. 
    data = defaultdict()
    
    if extended_features:
        prefix = 'X_extended_features_'
    else: 
        prefix = 'X_features_'
    files = [prefix + split for split in splits]
    baseline_files = ['baselines_' + split + '.pkl' for split in splits]
    
    for split, filename, baseline in zip(splits, files, baseline_files):
        filepath = os.path.join(path, filename + '.pkl')
        full_data = load_pickle(filepath)
        full_data = handle_index(full_data, index)
        y = full_data[label]
        X = full_data.drop(label, axis=1)
        if drop_col in X.columns:
            X = X.drop(columns=drop_col, axis=1)
            print(f'Shape after dropping {drop_col}: {X.shape}')    
        data[f'X_{split}'] = X
        data[f'y_{split}'] = y

        # Additionally, try to load baseline scores (if not physionet data):
        if not 'physionet2019' in path:
            baseline_path = os.path.join(path, baseline)
            baselines = load_pickle(baseline_path)
            baselines = handle_index(baselines, index)
            baselines = baselines.drop(label, axis=1)
            data[f'baselines_{split}'] = baselines 
        
    return data

def to_tsfresh_form(df, labels):
    """ Puts the data into a form where tsfresh extractors can be used. """
    tsfresh_frame = df.reset_index()
    labels = labels.groupby('id').apply(lambda x: x.iloc[0]).astype(int)
    return tsfresh_frame, labels
 
 
