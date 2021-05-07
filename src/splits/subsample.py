"""Script to generate subsampling splits for equalizing prevalences"""

import argparse
import numpy as np
import pandas as pd
import glob
import pathlib
import os
import sys
import time
import json
from IPython import embed
#from src.variables.mapping import VariableMapping
#
#VM_CONFIG_PATH = str(
#    pathlib.Path(__file__).parent.parent.parent.joinpath(
#        'config/variables.json'
#    )
#)
#VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)

def compute_subsamples(ids, labels, n_subsamplings, target_prev, seed=123):
    """
    Actually draw subsamples.
    """
    np.random.seed(seed)
    df = pd.DataFrame({'ids':ids, 'labels':labels})
    # disentangle cases and controls:
    cases = df[df['labels'] == 1]
    controls = df[df['labels'] == 0]
    # determine current prevalence of this split:
    curr_prev = len(cases) / len(df)
    if curr_prev > target_prev:
        print(f'Current prevalence {curr_prev} > target prev {target_prev}, subsampling cases..')
        n_select = (len(controls)*target_prev) / (1-target_prev)
        id_pool = cases['ids']
        untouched = controls['ids']
    elif curr_prev < target_prev:
        print(f'Current prevalence {curr_prev} < target prev {target_prev}, subsampling controls..')
        n_select = (len(cases)*(1-target_prev)) / target_prev 
        id_pool = controls['ids']
        untouched = cases['ids']
    else:
        raise ValueError('No subsampling needed.')

    # actually draw subsamples:
    d = {}
    df = df.set_index('ids') 
    for i in np.arange(n_subsamplings):
        assert n_select < len(id_pool)
        n_select = int(np.round(n_select))
        selected_ids = np.random.choice(id_pool, n_select)  
        # adding together subsampled plus unmodified ids (of opposing class)  
        tot = np.concatenate([selected_ids, untouched.values])
        curr_df = df.loc[tot].reset_index()
        new_prev = curr_df['labels'].sum() / len(curr_df)
        if (new_prev > target_prev*1.2) or (new_prev < target_prev*0.8):
            embed()

        print(f'Rep {i}, new prev: {new_prev}') 
        d[f'rep_{i}'] = {
            'ids': curr_df['ids'].tolist(), 
            'labels': curr_df['labels'].tolist()
        }
    # check coverage if cases were subsampled:
    if curr_prev > target_prev:
        remaining_cases = cases['ids'].tolist()
        n_cases_start = len(remaining_cases)
        n_cases_end = n_cases_start
        for i in np.arange(n_subsamplings):
            curr = d[f'rep_{i}']['ids']  
            for id in curr:
                if id in remaining_cases:
                    n_cases_end -= 1 
                    remaining_cases.remove(id)
        coverage = 1 - n_cases_end / n_cases_start
        print(f'cases start = {n_cases_start}, end = {n_cases_end}, coverage = {coverage}')
    return d
 
def get_subsamples(d, n_subsamplings, n_train_splits, target_prev):
    """
    Call subsampling routine for val/test splits.
    Args:
    -d: input dictionary with split ids
    -n_subsamplings: number of subsamples to draw
    -n_train_splits: number of train/val splits that are used
    -target_prev: target prevalence to subsample for 
    """
    
    ids = d['total']['ids']
    labels = d['total']['labels']
    id_label_map = {id: label for id, label in zip(ids, labels)}
   
    # 1. process validation splits
    for i in np.arange(n_train_splits):
        curr_ids = d['dev'][f'split_{i}']['validation']
        curr_labels = [id_label_map[x] for x in curr_ids]
        subsamples = compute_subsamples(
            curr_ids, 
            curr_labels,
            n_subsamplings,
            target_prev
        )
        d['dev'][f'split_{i}']['validation_subsamples'] = subsamples
 
    # 2. process test splits
    # total test set:
    curr_ids = d['test']['total']
    curr_labels = [id_label_map[x] for x in curr_ids]
    subsamples = compute_subsamples(
        curr_ids, 
        curr_labels,
        n_subsamplings,
        target_prev
    )
    d['test'][f'total_subsamples'] = subsamples
    # boosted test sets: 
    # we assume the same number as train splits, assert this:
    assert len(d['test']) == 2 + n_train_splits  
    for i in np.arange(n_train_splits):
        curr_ids = d['test'][f'split_{i}']
        curr_labels = [id_label_map[x] for x in curr_ids]
        subsamples = compute_subsamples(
            curr_ids, 
            curr_labels,
            n_subsamplings,
            target_prev
        )
        d['test'][f'split_{i}_subsamples'] = subsamples
   
    return d
  
    
def main(args):

    #Unpack arguments:
    in_path = args.in_path
    out_path = args.out_path
    dataset = args.dataset
    n_subsamplings = args.n_subsamplings
    n_train_splits = args.n_train_splits #n of train/val splits
    target_prev = args.target_prev
 
    if dataset == 'all':
        datasets = ['mimic', 'hirid', 'eicu', 'aumc', 'physionet2019']
    else:
        datasets = [dataset]

    # Loop over datasets:
    for dataset in datasets:
        
        in_file = os.path.join(in_path, f'splits_{dataset}.json')
        with open(in_file, 'r') as f:
            d = json.load(f)
    
        print(f'Processing {dataset}')
        d = get_subsamples(
            d,
            n_subsamplings, 
            n_train_splits,
            target_prev
        )  
     
        out_file = os.path.join(in_path, f'subsamples_{dataset}.json') 
        with open(out_file, 'w') as f:
            json.dump(d, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', default='config/splits', 
     help='Name of input file storing split information')
    parser.add_argument('--out_path', default='config/splits', 
     help='Name of output file storing subsampled split information')
    parser.add_argument('--dataset', default='all', 
     help='which dataset to process [mimic, hirid, eicu, aumc, physionet2019, all]')
    parser.add_argument('--n_subsamplings', type=int, default=10, 
     help='Number of subsamplings to create.')
    parser.add_argument('--n_train_splits', type=int, default=5, 
     help='Number of stratified shuffle splits train/val to generate')
    parser.add_argument('--target_prev', type=float, default=0.17, 
     help='Target prevalence of each subsampled cohort')

    args = parser.parse_args()

    main(args)
