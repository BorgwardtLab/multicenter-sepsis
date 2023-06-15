import json
import fire
import os
import sys
from tqdm import tqdm
sys.path.append('../..')
from patient_evaluation import main as patient_eval
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import auc 
from scipy import interpolate
import pandas as pd

def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def main(
    eval_folder = 'evaluation_pairwise_bootstraps', #input 
    out_file='roc_bootstrap.csv',
    ):
   
    # loop over all configurations to compute AUC from evaluation file:

    datasets = ['aumc', 'hirid', 'mimic', 'eicu']
    n_reps = 5; n_subsamples = 10
    n_bootstraps = 40

    configs = [(dataset_train, dataset_eval, rep, subsample)
            for dataset_train in datasets
            for dataset_eval in datasets
            for rep in range(n_reps)
            for subsample in range(n_subsamples)]

    bt_auc = pd.DataFrame() # gathering all bootstraps in the inner loop
    mean_fpr = np.linspace(0, 1, 200)
    for c in tqdm(configs):
        dataset_train, dataset_eval, rep, subsample = c 
        # read current file:
        eval_file = os.path.join(
            eval_folder,
            f'bootstrap_eval_{c[0]}_{c[1]}_rep_{c[2]}_subsample_{c[3]}.json'
        )
        evals = load_json(eval_file)
        for i in range(n_bootstraps):
            # go over boostraps:
            d = evals[i]
            df = pd.DataFrame(d) 

            tpr = df['pat_recall'].values
            fpr = 1 - df['pat_specificity'].values
            tpr = np.append(np.append([1], tpr), [0])
            fpr = np.append(np.append([1], fpr), [0])
            fn = interpolate.interp1d(fpr, tpr) #interpolation fn
            interp_tpr = fn(mean_fpr)
            #tprs.append(interp_tpr)

            # Inside the loop, gather all auc / ROC results as bootstraps:
            curr_auc = auc(mean_fpr, interp_tpr) 
            curr_boot_df = pd.DataFrame(
                {   'AUC': [curr_auc], 
                    'rep': [rep], 
                    'subsample': [subsample],
                    'bootstrap': [i], 
                    'model': ['AttentionModel'],
                    'train_dataset': [dataset_train],
                    'eval_dataset': [dataset_eval]
                }
            )
            # bootstraps df with raw ROC entries:
            bt_auc = bt_auc.append(curr_boot_df)
    bt_auc.to_csv(out_file)

if __name__ == "__main__":

    fire.Fire(main)
