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

def model_map(name):
    if name == 'AttentionModel':
        name = 'attention model'
    return name

def main():
    models = ['AttentionModel', 'lgbm', 'sofa', 'qsofa'] #'sirs', 'mews', 'news',
    datasets = ['aumc', 'hirid', 'eicu', 'mimic']

    infile ='results/evaluation/plots/result_data.csv'
    df = pd.read_csv(infile)
    for train_dataset in datasets:
        for eval_dataset in datasets:
            plt.figure()
            for model in models:
                filter_dict = {
                    'model': model,
                    'dataset_train': train_dataset,
                    'dataset_eval':  eval_dataset
                }
                data = df_filter(df, filter_dict) 
                if len(data) < 1:
                    continue 
                reps = data['rep'].unique()
                
                #TODO: add sanity check that data has 1000 rows, and 5 different reps
                tprs = []
                aucs = []
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
                    aucs.append(roc_auc)
                    curr_df = pd.DataFrame(
                        { 'fpr': mean_fpr,
                          'tpr': interp_tpr}
                    )
                    curr_df['rep'] = rep
                    metrics = metrics.append(curr_df)
                aucs = np.array(aucs)
                auc_mean = aucs.mean()
                auc_std = aucs.std()
                sns.lineplot(data=metrics, x="fpr", y="tpr", 
                    label=model_map(model) + rf', AUROC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')
            if train_dataset == eval_dataset: 
                title=f'ROC Curve for internal validation on {train_dataset}'
            else: 
                title = f'ROC Curve for external validation: trained on {train_dataset}, tested on {eval_dataset}'
            plt.title(title) 
            plt.legend(loc='lower right') 
            plt.savefig(f'results/evaluation/plots/roc_{train_dataset}_{eval_dataset}.png') 
if __name__ == '__main__':
    main()
