"""Script to generate stratified splits"""

import argparse
import numpy as np
import pandas as pd
import glob
import pathlib
import os
import sys
import time
import json
from sklearn.model_selection import StratifiedShuffleSplit
from IPython import embed
from src.datasets.utils import get_file_mapping
from src.variables.mapping import VariableMapping

VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)
VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


def main(args):
    #Unpack arguments:
    n_train_splits = args.n_train_splits
    test_size = args.test_size
    validation_size = args.validation_size
    n_test_splits = args.n_test_splits
    test_boost_size = args.test_boost_size 
    path = args.path
    out_path = args.out_path
    dataset = args.dataset
    version = args.version
    file_mapping = get_file_mapping(version)
    adjusted_val_size = validation_size / ( 1 - test_size)  

    if dataset == 'all':
        datasets = file_mapping.keys()
    else:
        datasets = [dataset]

    # Loop over datasets:
    for dataset in datasets:
        
        # Filter out invalid ids:
        filters = None
        #if dataset == 'mimic':
        #    filtered_ids = [227023, 246137, 253670]
        #    print(f'>>> Manually filtering the following ids: {filtered_ids}')
        #    filters = [(VM_DEFAULT('id'), 'not in', filtered_ids)] 
        # Load data 
        df = pd.read_parquet( 
            os.path.join( path, file_mapping[dataset]),
            engine='pyarrow',
            filters=filters
        )
        d = {}
        df = df.set_index(VM_DEFAULT('id'))
        ids = np.array(df.index.unique())
        try: 
            labels = np.array(
                [ df.loc[i][VM_DEFAULT('label')].any() * 1 for i in ids ]
            )
        except:
            from IPython import embed; embed()
        # we write arrays (easier to subset) and write them out as lists (easier with json) 
        # first gather *all* ids and labels under 'total'
        d['total'] = {'ids': ids.tolist(), 'labels': labels.tolist()}
        
        # 1. we first split once into development / test split:
        d['dev'] = {}; d['test'] = {}
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        sss.get_n_splits(labels)
        for dev_index, test_index in sss.split(labels,labels):
            dev_ids = ids[dev_index]
            test_ids = ids[test_index]
            d['dev']['total'] = dev_ids.tolist()
            d['test']['total'] = test_ids.tolist() 
       
        # 2. we split the development split into train / val and this n_train_split times
        # adjust val_size due to having already removed test data
        
        print(f'validation size of {validation_size} results in an adjusted val size of {adjusted_val_size}')
        sss_tv = StratifiedShuffleSplit(n_splits=n_train_splits, test_size=adjusted_val_size, random_state=42)
        
        dev_labels = labels[dev_index] 
        sss_tv.get_n_splits(dev_labels)
        for counter, (tr_ind, val_ind) in enumerate(sss_tv.split(dev_labels, dev_labels)):
            train_ids = dev_ids[tr_ind]
            val_ids = dev_ids[val_ind]
            current_split = {'train': train_ids.tolist(), 
                             'validation': val_ids.tolist(),
            }
            d['dev'][f'split_{counter}'] = current_split
        
        # 3. we split the test split into n_test_splits splits for varying the test data.
        test_size = 1-test_boost_size # if we want 80% of the data, use test_size 0.2 
        sss_te = StratifiedShuffleSplit(n_splits=n_test_splits, test_size=test_size, random_state=42)
        test_labels = labels[test_index]
        sss_te.get_n_splits(test_labels)
        for counter, (te_ind, _) in enumerate(sss_te.split(test_labels, test_labels)):
            te_ids = test_ids[te_ind]
            d['test'][f'split_{counter}'] = te_ids.tolist()

        #4. Dump split info to json file:
        #--------------------------------
        d['version'] = version 
        print('Dumping split info to json...')
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, f'splits_{dataset}_v{version}.json') 
        with open(out_file, 'w') as f:
            json.dump(d, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets/downloads', 
     help='Path to data dir')
    parser.add_argument('--out_path', default='config/splits', 
     help='Name of output file storing split information')
    parser.add_argument('--dataset', default='all', 
     help='which dataset to process [mimic, hirid, eicu, aumc, physionet2019, all]')
    parser.add_argument('--n_train_splits', type=int, default=5, 
     help='Number of stratified shuffle splits train/val to generate')
    parser.add_argument('--test_size', type=float, default=0.1, 
     help='Ratio of patients in test split')
    parser.add_argument('--validation_size', type=float, default=0.1, 
     help='Ratio of patients in validation split')
    # for boosting test splits:
    parser.add_argument('--n_test_splits', type=int, default=5, 
    help='Number of stratified shuffle test splits to generate')
    parser.add_argument('--test_boost_size', type=float, default=0.8, 
     help='Ratio of patients in test boosting splits')
    parser.add_argument('--version', default='0.3.0', 
     help='data version')


    args = parser.parse_args()

    main(args)
