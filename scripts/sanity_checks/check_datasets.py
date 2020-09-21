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
    hours = 0 
    for index, instance in enumerate(d):
        new_dim = instance['ts'].shape[1]
        t = instance['ts'].shape[0]
        hours += t
        if len(dims) == 0:
            dims.append(new_dim) 
        elif new_dim not in dims:
            print(f'Found new feature dim {new_dim} at index {index}, previous dim was {dim}')
            dim_change = True
            change_indices.append(index)
            dims.append(new_dim)
        
    return dims, dim_change, change_indices, hours


if __name__ == "__main__":
    outpath = 'scripts/sanity_checks' 
    datasets = src.datasets.__all__
    splits = ['train', 'validation', 'test']
    results = {}
    total_hours = 0
    for dataset in datasets:
        print(f'Checking {dataset} ..')
        results[dataset] = {}
        dataset_hours = 0 
        for split in splits:
            print(f'{split} split:')
            results[dataset][split] = {}

            dataset_cls = getattr(src.datasets, dataset)
            d = dataset_cls(split=split)
            dims, dim_change, change_indices, hours = check_feature_dim_of_dataset(d)
            dataset_hours += hours 
 
            results[dataset][split]['dims'] = dims
            results[dataset][split]['dim_change'] = dim_change
            results[dataset][split]['change_indices'] = change_indices
        print(f'{dataset} has {dataset_hours} hours.')
        total_hours += dataset_hours
    print(f'Total hours: {total_hours}')
    with open(os.path.join(outpath, 'dim_check.pkl'), 'wb') as f:
        pickle.dump(results, f)

