import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import auc 
from scipy import interpolate

from IPython import embed
import sys

def df_filter(df, filter_dict):
    for key, val in filter_dict.items():
        df = df[df[key] == val]
    return df

def main():
    models = ['lgbm', 'sofa', 'sirs', 'mews', 'news','qsofa']
    datasets = ['aumc', 'hirid', 'eicu', 'mimic']
    infile ='results/evaluation/plots/result_data.csv'
    df = pd.read_csv(infile)
    plt.figure()
    for dataset in datasets:
        for model in models:
            filter_dict = {
                'model': model,
                'dataset_train': dataset,
                'dataset_eval': dataset
            }
            data = df_filter(df, filter_dict) 
            if len(data) < 1:
                continue 
            reps = data['rep'].unique()
            
            #TODO: add sanity check that data has 1000 rows, and 5 different reps
            tprs = []
            aucs = {}
            mean_fpr = np.linspace(0, 1, 200)
            metrics = pd.DataFrame()
            for rep in reps:
                rep_data = data[data['rep'] == rep]
                tpr = rep_data['pat_recall'].values
                fpr = 1 - rep_data['pat_specificity'].values
                tpr = np.append(np.append([1], tpr), [0])
                fpr = np.append(np.append([1], fpr), [0])
                fn = interpolate.interp1d(fpr, tpr) #interpolation fn
                interp_tpr = fn(mean_fpr)
                #interp_tpr = np.interp(mean_fpr, fpr, tpr.values)
                #interp_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr) #on raw values
                interp_auc = auc(mean_fpr, interp_tpr) #on raw values
                tprs.append(interp_tpr)
                aucs[rep] = roc_auc
                curr_df = pd.DataFrame(
                    { 'fpr': mean_fpr,
                      'tpr': interp_tpr}
                )
                curr_df['rep'] = rep
                metrics = metrics.append(curr_df)
            sns.lineplot(data=metrics, x="fpr", y="tpr", label=model)
        plt.title(f'ROC Curve for {dataset}') 
        plt.legend(loc='lower right') 
        plt.savefig(f'test_runs/roc_{dataset}.png') 
if __name__ == '__main__':
    main()
