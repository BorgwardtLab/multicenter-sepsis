"""Script to generate stratified splits for Physionet 2019"""
"""Author: Michael Moor"""

import argparse
import numpy as np
import pandas as pd
import glob
import os
import sys
import time
import pickle

from sklearn.model_selection import StratifiedShuffleSplit
from IPython import embed

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def test(info, n_splits=5):
    for counter in np.arange(n_splits):
        test = info[f'split_{counter}']['test']
        train = info[f'split_{counter}']['train']
        val = info[f'split_{counter}']['validation']
        i1 = intersection(test, train)
        i2 = intersection(test, val)
        i3 = intersection(val, train)
        for inter in [i1, i2, i3]:
            if len(inter) > 0:
                print('Non empty intersection within split found!')
                print(f'Split {counter}')

#hard-coded path for prototyping:
def main(args):
    #Unpack arguments:
    n_splits = args.n_splits
    test_size = args.test_size
    validation_size = args.validation_size

    path = "data/extracted/*.psv"
    info = {'pat_ids': [], 'labels': []}

    #1. Iterate over patient files:
    #------------------------------    
    print('Iterating over patient files..')
    for i, fname in enumerate(glob.glob(path)):
        pat = pd.read_csv(fname, sep='|')

        # get patient IDs and Labels
        head, tail = os.path.split(fname)
        #pat_id = int(tail[1:7]) #hard-coded indexing (works for physionet)
        pat_id = int(tail.lstrip('p').rstrip('.psv') )

        label = 1 if pat['sep3'].sum() > 0 else 0

         # append current patient info to dict:
        info['pat_ids'].append(pat_id)
        info['labels'].append(label)

    #gather pat_ids and labels in np array for easy list subsetting:
    info['pat_ids'] = np.array(info['pat_ids'])
    info['labels'] = np.array(info['labels'])

    #2. Create n splits (train,val,test)
    #-----------------------------------
    print(f'Creating {n_splits} splits ...')
    y = info['labels'] # nicer name
    #first split into train / test:
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    sss.get_n_splits(y)
    # create 1 additional train/val split from train split:
    # to compute test_size for validation split, need to take train size of above split into account
    adjusted_val_size = validation_size / ( 1 - test_size)  
    print(f'validation size of {validation_size} results in an adjusted val size of {adjusted_val_size}')
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=42)

    # Iterate over both splitting steps to assign ids to splits 
    for counter, (train_index, test_index) in enumerate(sss.split(y,y)):
        #print("TRAIN:", train_index, "TEST:", test_index)
        #print("train IDs:", info['pat_ids'][train_index])
        info[f'split_{counter}'] = {'train': [], 'validation': [], 'test': []}
        #directly assign test ids:
        info[f'split_{counter}']['test'] = info['pat_ids'][test_index]
        #further divide train into train and val:
        y_train = info['labels'][train_index]
        sss_val.get_n_splits(y_train)
        for tr_ind, val_ind in sss_val.split(y_train, y_train):
            info[f'split_{counter}']['train'] = info['pat_ids'][train_index][tr_ind]
            info[f'split_{counter}']['validation'] = info['pat_ids'][train_index][val_ind]     

    #3. Sanity Check: Is there no overlap of ids between train/val/test of a given split?
    #------------------------------------------------------------------------------------ 
    print('Running test (intersection between splits)')
    test(info)

    #4. Dump split info into pkl file:
    #--------------------------------
    print('Dumping split info into pickle...')
    with open('data/split_info.pkl', 'wb') as f:
        pickle.dump(info, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_splits', type=int, default=5, 
     help='Number of stratified shuffle splits to generate')
    parser.add_argument('--test_size', type=float, default=0.1, 
     help='Ratio of patients in test split')
    parser.add_argument('--validation_size', type=float, default=0.1, 
     help='Ratio of patients in validation split')
    args = parser.parse_args()

    main(args)
