import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import auc 
from scipy import interpolate
import argparse
import os
from IPython import embed
import sys

def df_filter(df, filter_dict):
    for key, val in filter_dict.items():
        df = df[df[key] == val]
    return df

def model_map(name):
    if name == 'AttentionModel':
        name = 'attn'
    elif name == 'GRUModel':
        name = 'gru'

    # harmonize str length; adjust as we see fit
    return name.ljust(6, ' ')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        required=True,
        type=str,
        help='Path to result dataframe file',
    )
    args = parser.parse_args()
    input_path = args.input_path
    models = ['AttentionModel', 'GRUModel', 'lgbm', 'lr', 'sofa', 'qsofa', 'sirs', 'mews', 'news'] #, 'news',
    datasets = ['physionet2019', 'aumc', 'hirid', 'eicu', 'mimic']

    #infile ='results/evaluation/plots/result_data.csv'
    df = pd.read_csv(input_path)
    use_subsamples = True if 'subsample' in df.columns else False
    if use_subsamples:
        n_subsamples = df['subsample'].unique().shape[0]
    else:
        n_subsamples = 1 # to mimic subsampling

    summary = []

    sns.set(font='Helvetica')

    for (train_dataset, eval_dataset), df_ in df.groupby(['dataset_train', 'dataset_eval']):
        print(train_dataset)

        plt.figure()
        for (model), data in df_.groupby(['model']): #same ordering as scatter plot
            #for train_dataset in datasets:
            #    for eval_dataset in datasets:
            #        plt.figure()
            #        for model in models:
            #            filter_dict = {
            #                'model': model,
            #                'dataset_train': train_dataset,
            #                'dataset_eval':  eval_dataset
            #            }
            #            data = df_filter(df, filter_dict) 
            if len(data) < 1:
                continue 
            reps = data['rep'].unique()
            
            aucs = []
            mean_fpr = np.linspace(0, 1, 200)
            metrics = pd.DataFrame()
           
            # loop over (potential) subsamples and repetition folds:
            for rep in reps:
                tprs = []
                for subsample in np.arange(n_subsamples):
                    if use_subsamples:
                        rep_filter = {'rep': rep, 'subsample': subsample}
                    else:
                        rep_filter = {'rep': rep}
                    rep_data = df_filter(data, rep_filter)

                    tpr = rep_data['pat_recall'].values
                    fpr = 1 - rep_data['pat_specificity'].values
                    tpr = np.append(np.append([1], tpr), [0])
                    fpr = np.append(np.append([1], fpr), [0])
                    fn = interpolate.interp1d(fpr, tpr) #interpolation fn
                    interp_tpr = fn(mean_fpr)
                    tprs.append(interp_tpr)
                    #interp_tpr = np.interp(mean_fpr, fpr, tpr.values)
                    #interp_tpr[0] = 0.0
                mean_tpr = np.mean(tprs, axis=0)
                roc_auc = auc(mean_fpr, mean_tpr) #on raw values
                aucs.append(roc_auc)
                curr_df = pd.DataFrame(
                    { 'False positive rate': mean_fpr,
                      'True positive rate': mean_tpr}
                )
                curr_df['rep'] = rep
                metrics = metrics.append(curr_df)

            aucs = np.array(aucs)
            auc_mean = aucs.mean()
            auc_std = aucs.std()
            sns.lineplot(
                data=metrics,
                x="False positive rate",
                y="True positive rate", 
                label=model_map(model) +'\t' + rf'AUROC = {auc_mean:.3f} $\pm$ {auc_std:.3f}',
            )
                # [model_map(model),'AUROC = ', f'{auc_mean:.3f}' + r' $\pm$ ' + f'{auc_std:.3f}'])
                # model_map(model) + rf' AUROC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')

            summary_df = pd.DataFrame(
                {
                    'model': [model],
                    'train_dataset': [train_dataset],
                    'eval_dataset': [eval_dataset],
                    'auc_mean': [auc_mean],
                    'auc_std': [auc_std]
                }
            )
            summary.append(summary_df)
        if train_dataset == eval_dataset: 
            title=f'ROC Curve for internal validation on {train_dataset}'
        else: 
            title = f'ROC Curve for external validation: trained on {train_dataset}, tested on {eval_dataset}'
        plt.title(title) 
        plt.legend(loc='lower right') #, ncol = 2)
        outfile = f'roc_{train_dataset}_{eval_dataset}'
        if 'subsampled' in os.path.split(input_path)[-1]:
            outfile += '_subsampled'
        outfile = os.path.join(os.path.split(input_path)[0], outfile + '.png') 
        plt.savefig(outfile, dpi=300)

    summary = pd.concat(summary)
    summary_file = os.path.join(os.path.split(input_path)[0], 'roc_summary') 
    if 'subsampled' in os.path.split(input_path)[-1]:
        summary_file += '_subsampled'
    summary.to_csv(summary_file + '.csv')
         
if __name__ == '__main__':
    main()
