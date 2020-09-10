""" 
This script checks all datasets, returns their feature dimensions 
as some categorical features are not harmonized yet.
"""

import src.datasets
import pickle
import os 

def check_feature_dim_of_dataset(d):
    dim_change = False
    change_indices = []
    dims = [] 
    for index, instance in enumerate(d):
        new_dim = instance['ts'].shape[1]
        if len(dims) == 0:
            dims.append(new_dim) 
        elif new_dim not in dims:
            print(f'Found new feature dim {new_dim} at index {index}, previous dim was {dim}')
            dim_change = True
            change_indices.append(index)
            dims.append(new_dim)
    return dims, dim_change, change_indices



if __name__ == "__main__":
    outpath = 'scripts/sanity_checks' 
    datasets = src.datasets.__all__
    splits = ['train', 'validation']
    results = {}
    for dataset in datasets:
        print(f'Checking {dataset} ..')
        results[dataset] = {}
        for split in splits:
            print(f'{split} split:')
            results[dataset][split] = {}

            dataset_cls = getattr(src.datasets, dataset)
            d = dataset_cls(split=split)
            dims, dim_change, change_indices = check_feature_dim_of_dataset(d)
            
            results[dataset][split]['dims'] = dims
            results[dataset][split]['dim_change'] = dim_change
            results[dataset][split]['change_indices'] = change_indices
    with open(os.path.join(outpath, 'dim_check.pkl'), 'wb') as f:
        pickle.dump(results, f)

