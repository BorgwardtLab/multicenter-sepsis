import scipy 
import numpy as np
import pandas as pd
from IPython import embed

def main():
    # hard coded result files:
    f1 = 'results/evaluation_test/plots/roc_pooled_bootstrap_subsampled.csv'
    f2 = 'results/evaluation_test/plots/roc_bootstrap_subsampled.csv'
    dfs = [pd.read_csv(f) for f in (f1, f2)]
    
    # check to remove redundant index column
    drop_col = 'Unnamed: 0'
    dfs_new = []
    for df in dfs:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
        dfs_new.append(df)
        print(df.columns)
    dfs = dfs_new

    # join pooled and single dataset results
    df = pd.concat(dfs)
    datasets = ['aumc', 'eicu', 'hirid', 'mimic']

    # emory is reported seperately:
    df = df.query("train_dataset != 'emory' & eval_dataset != 'emory'")
    
    # for each evaluation site (core datasets),
    # get paired results: pooled vs mean of single dataset perf.
    # firsts, get pooled results:
    df_p = df.query("train_dataset == 'pooled'")
    # get single dataset results:
    df_s = df.query("train_dataset != 'pooled' & train_dataset != eval_dataset")

    embed() 
    
if __name__ == "__main__":
    main()
