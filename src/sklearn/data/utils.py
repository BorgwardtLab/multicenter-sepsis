import os
import pickle
import json

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



