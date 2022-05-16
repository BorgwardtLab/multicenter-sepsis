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

def emory_map(name):
    if name == 'physionet2019':
        name = 'emory'
    return name 

def raw_to_csv(metrics, csv_path, auc_mean, auc_std):
    """ write raw roc values to csv"""
    cols = [col for col in metrics.columns if not 'rep' in col]
    out = {}
    for col in cols:
        curr_df = metrics[[col, 'rep']]
        piv = curr_df.pivot(columns='rep')
        mu = piv.mean(axis=1)
        sig = piv.std(axis=1)
        out[col + '_mean'] = mu
        out[col + '_std'] = sig
    df = pd.DataFrame(out)
    df['auc_mean'] = auc_mean
    df['auc_std'] = auc_std
    df.to_csv(csv_path, index=False)
    
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
    #models = ['AttentionModel', 'GRUModel', 'lgbm', 'lr', 'sofa', 'qsofa', 'sirs', 'mews', 'news'] #, 'news',
    #datasets = ['physionet2019', 'aumc', 'hirid', 'eicu', 'mimic']

    #infile ='results/evaluation/plots/result_data.csv'
    df = pd.read_csv(input_path)
    for col in ['dataset_train','dataset_eval']:
        df[col] = df[col].apply(emory_map)

    use_subsamples = True if 'subsample' in df.columns else False
    if use_subsamples:
        n_subsamples = df['subsample'].unique().shape[0]
    else:
        n_subsamples = 1 # to mimic subsampling

    summary = []
    bt_auc = pd.DataFrame() # gathering all bootstraps in the inner loop

    #sns.set(font='Helvetica')

    output_path = os.path.split(input_path)[0]

    for (train_dataset, eval_dataset), df_ in df.groupby(['dataset_train', 'dataset_eval']):
        print(train_dataset)
        
        plt.figure()
        for (model), data in df_.groupby(['model']): #same ordering as scatter plot
            
            if len(data) < 1:
                continue 
            reps = data['rep'].unique()
            
            aucs = []
            mean_fpr = np.linspace(0, 1, 200)
            metrics = pd.DataFrame() # gathering metrics over repetition splits
            
 
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

                    # Inside the loop, gather all auc / ROC results as bootstraps:
                    curr_auc = auc(mean_fpr, interp_tpr) 
                    curr_boot_df = pd.DataFrame(
                        {   'AUC': [curr_auc], 
                            'rep': [rep], 
                            'subsample': [subsample],
                            'model': [model],
                            'train_dataset': [train_dataset],
                            'eval_dataset': [eval_dataset]
                        }
                    )
                    # bootstraps df with raw ROC entries:
                    bt_auc = bt_auc.append(curr_boot_df)

                # Means over subsampling, for each repetition split
                mean_tpr = np.mean(tprs, axis=0)
                roc_auc = auc(mean_fpr, mean_tpr) #on raw values
                aucs.append(roc_auc)
                curr_df = pd.DataFrame(
                    { '1 - Specificity': mean_fpr,
                      'Sensitivity': mean_tpr}
                )
                curr_df['rep'] = rep
                metrics = metrics.append(curr_df)

            aucs = np.array(aucs)
            auc_mean = aucs.mean()
            auc_std = aucs.std()
            
            metrics.reset_index(drop=True, inplace=True) # sns doesn't like duplicate indices
            sns.lineplot(
                data=metrics,
                x="1 - Specificity",
                y="Sensitivity", 
                label= '{:<8}'.format(model_map(model)) + rf'AUC = {auc_mean:.3f} $\pm$ {auc_std:.3f}',
            )
                #label=model_map(model) +'\t' + rf'AUROC = {auc_mean:.3f} $\pm$ {auc_std:.3f}',

                # [model_map(model),'AUROC = ', f'{auc_mean:.3f}' + r' $\pm$ ' + f'{auc_std:.3f}'])
                # model_map(model) + rf' AUROC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')
            
            # write raw roc data to csv:
            csv_path = os.path.join(output_path, f'raw_roc_data_{model}_{train_dataset}_{eval_dataset}.csv')
            raw_to_csv(metrics, csv_path, auc_mean, auc_std) 

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
        plt.legend(loc='lower right', prop={'family': 'monospace'}) #, ncol = 2)
        outfile = f'roc_{train_dataset}_{eval_dataset}'
        if 'subsampled' in os.path.split(input_path)[-1]:
            outfile += '_subsampled'
        outfile = os.path.join(output_path, outfile + '.png') 
        plt.savefig(outfile, dpi=300)

    # Summary aggregated over subsamples, variation in repetition splits:
    summary = pd.concat(summary)
    summary_file = os.path.join(output_path, 'roc_summary') 
    if 'subsampled' in os.path.split(input_path)[-1]:
        summary_file += '_subsampled'
    summary.to_csv(summary_file + '.csv')
    
    # Bootstrap results (inner loop) also showing subsampling variation:
    bt_file = os.path.join(output_path, 'roc_bootstrap') 
    if 'subsampled' in os.path.split(input_path)[-1]:
        bt_file += '_subsampled'
    bt_auc.to_csv(bt_file + '.csv')
    embed()
         
if __name__ == '__main__':
    main()
