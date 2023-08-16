import pandas as pd
import os
from IPython import embed
from tqdm import tqdm 
import fire
import json
import random
import numpy as np
from joblib import Parallel, delayed

def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def get_bootstraps(n, n_bootstraps=40, seed=42):
    """
    n: sample size
    n_bootstraps: number of bootstraps
    returns list of indices (with repetition)
    """
    random.seed(seed)
    bootstrap_samples = [random.choices(range(n), k=n) for _ in range(n_bootstraps)]
    return bootstrap_samples

def apply_subset(d, indices, bootstrap_counter=0):
    """
    Subset prediction dict d (similar to subsampling code).
    """

    # keys (with patient-level information) to subset:
    keys = ['ids', 'labels', 'targets', 'predictions', 'scores', 'times'] 
    
    
    #mask = np.isin(d['ids'], subsample_ids) 
    output = d.copy()
    for key in keys:
        output.pop(key)
        as_array = np.array(d[key], dtype=object)
        output[key] = as_array[indices].tolist()

    output['bootstrap_sample'] = bootstrap_counter

    return output


def main(n_bootstraps=40,
    out_path = 'pairwise_bootstraps',
    disk_root = '/lfs/trinity/0/mdmoor/mcsep',  #/Volumes/Mephisto/PhD/multicenter-sepsis/multicenter-sepsis/',
    pred_mapping_file = 'prediction_subsampled_mapping.json', 
    ):
    # we want to filter to for sepsis cases or controls.
    os.makedirs(out_path, exist_ok=True)

    pred_mapping_file = os.path.join(disk_root, pred_mapping_file) 

    # load data:
    #preds = load_json(pred_file)
    pred_mapping = load_json(pred_mapping_file)
    pred_mapping = pred_mapping['AttentionModel']
    
    datasets = ['aumc', 'hirid', 'mimic', 'eicu']
    n_reps = 5; n_subsamples = 10

    # generate all arguments
    arguments = [(dataset_train, dataset_eval, rep, subsample, pred_mapping, n_bootstraps, out_path, disk_root) 
                for dataset_train in datasets 
                for dataset_eval in datasets 
                for rep in range(n_reps) 
                for subsample in range(n_subsamples)]

    # run in parallel
    n_jobs = 100  # set this to the number of cores you want to use, -1 means using all processors
    _ = Parallel(n_jobs=n_jobs)(delayed(process_sample)(*args) for args in arguments)


    #total = len(datasets)**2 * n_reps * n_subsamples
    #pbar = tqdm(total=total, desc="Progress")
    #for dataset_train in datasets:
    #    for dataset_eval in datasets:
    #        for rep in range(n_reps):
    #            for subsample in range(n_subsamples):

    #                #TODO: customize this for other than internal results: 
    #                pred_file = pred_mapping[dataset_train][dataset_eval][f'rep_{rep}'][f'subsample_{subsample}']
    #                preds = load_json(
    #                        os.path.join(disk_root, pred_file)
    #                )
    #                # get bootstrap indices (with repetition)
    #                bootstrap_indices = get_bootstraps(
    #                    n = len(preds['ids']),
    #                    n_bootstraps = n_bootstraps
    #                )
    #                bootstrap_preds = []
    #                # apply subset for each bootstrap:
    #                for i in range(n_bootstraps):
    #                    bt_pred = apply_subset(preds, bootstrap_indices[i], bootstrap_counter=i)
    #                    bootstrap_preds.append(bt_pred)
    #                
    #                out_file = os.path.join(
    #                        out_path,
    #                        f'bootstrap_pred_{dataset_train}_{dataset_eval}_rep_{rep}_subsample_{subsample}.json'
    #                )
    #                with open(out_file, 'w') as f:
    #                    json.dump(bootstrap_preds, f, indent=4)

    #                pbar.update()

    #pbar.close()

def process_sample(dataset_train, dataset_eval, rep, subsample, pred_mapping, n_bootstraps, out_path, disk_root):
    pred_file = pred_mapping[dataset_train][dataset_eval][f'rep_{rep}'][f'subsample_{subsample}']
    # fix path for server:
    old_path = 'revisions/results/evaluation_test'
    pred_file = pred_file.lstrip(old_path)

    preds = load_json(os.path.join(disk_root, pred_file))
    bootstrap_indices = get_bootstraps(n = len(preds['ids']), n_bootstraps = n_bootstraps)
    bootstrap_preds = [apply_subset(preds, bootstrap_indices[i], bootstrap_counter=i) for i in range(n_bootstraps)]

    out_file = os.path.join(
            out_path,
            f'bootstrap_pred_{dataset_train}_{dataset_eval}_rep_{rep}_subsample_{subsample}.json'
    )
    with open(out_file, 'w') as f:
        json.dump(bootstrap_preds, f, indent=4)
    return None  # we don't really need to return anything

if __name__ == "__main__":
    fire.Fire(main)

