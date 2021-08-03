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
import wandb

wandb_api = wandb.Api()

def get_feature_set(run_id, run_path='sepsis/mc-sepsis/sweeps'):
    """ map run_id to used feature set via wandb API"""
    run_path = os.path.join(run_path, run_id)
    run = wandb_api.run(run_path)
    run_info = run.config
    return run_info['dataset_kwargs/feature_set']

def get_run_to_feature_set_mapping(run_ids):
    d = {i: get_feature_set(i) for i in run_ids}
    def wrapped(run_id):
        return d[run_id]
    return wrapped

def get_run(fname):
    """get run_id from file name """
    return fname.split('/')[-1].split('_')[0]

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
    models = ['AttentionModel'] #, 'GRUModel', 'lgbm', 'lr', 'sofa', 'qsofa', 'sirs', 'mews', 'news'] #, 'news',
    datasets = ['mimic'] #['physionet2019', 'aumc', 'hirid', 'eicu', 'mimic']

    #infile ='results/evaluation/plots/result_data.csv'
    df = pd.read_csv(input_path)

    #extract run_id from each filename:
    df['run_id'] = df['fname'].apply(get_run)
    feature_set_map = get_run_to_feature_set_mapping(
        df['run_id'].unique()
    )
    # map run_id to feature set
    df['feature_set'] = df['run_id'].apply(feature_set_map) 

    use_subsamples = True if 'subsample' in df.columns else False
    if use_subsamples:
        n_subsamples = df['subsample'].unique().shape[0]
    else:
        n_subsamples = 1 # to mimic subsampling

    summary = []

    sns.set(font='Helvetica')

    output_path = os.path.split(input_path)[0]

    for (train_dataset, eval_dataset), df_ in df.groupby(['dataset_train', 'dataset_eval']):
        print(train_dataset)

        for (model), data_ in df_.groupby(['model']): #same ordering as scatter plot
       
            plt.figure() 
            for (feature_set), data in data_.groupby(['feature_set']):
                print(feature_set)
                if len(data) < 1:
                    print(f'len(data) = {len(data)}')
                    embed() 
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
                    label=feature_set +'\t' + rf'AUROC = {auc_mean:.3f} $\pm$ {auc_std:.3f}',
                )
                    # [model_map(model),'AUROC = ', f'{auc_mean:.3f}' + r' $\pm$ ' + f'{auc_std:.3f}'])
                    # model_map(model) + rf' AUROC = {auc_mean:.3f} $\pm$ {auc_std:.3f}')
                
                # write raw roc data to csv:
                csv_path = os.path.join(output_path, f'raw_roc_data_{model}_{train_dataset}_{eval_dataset}_{feature_set}.csv')
                raw_to_csv(metrics, csv_path, auc_mean, auc_std) 

                summary_df = pd.DataFrame(
                    {
                        'model': [model],
                        'train_dataset': [train_dataset],
                        'eval_dataset': [eval_dataset],
                        'auc_mean': [auc_mean],
                        'auc_std': [auc_std],
                        'feature_set': [feature_set]
                    }
                )
                summary.append(summary_df)
            title=f'ROC Curve {model_map(model)} model on {train_dataset}'
            plt.title(title) 
            plt.legend(loc='lower right') #, ncol = 2)
            outfile = f'roc_{train_dataset}_{eval_dataset}_{model}'
            if 'subsampled' in os.path.split(input_path)[-1]:
                outfile += '_subsampled'
            outfile = os.path.join(output_path, outfile + '.png') 
            plt.savefig(outfile, dpi=300)

    summary = pd.concat(summary)
    summary_file = os.path.join(output_path, 'roc_summary') 
    if 'subsampled' in os.path.split(input_path)[-1]:
        summary_file += '_subsampled'
    summary.to_csv(summary_file + '.csv')
         
if __name__ == '__main__':
    main()