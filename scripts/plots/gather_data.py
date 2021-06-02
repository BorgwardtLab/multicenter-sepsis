import glob
from IPython import embed
import os 
import json
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation.patient_evaluation import format_dataset 
import argparse
from joblib import Parallel, delayed


def make_scatter(df, fname, signature, title=None):
    fig = plt.figure()
    sns.scatterplot(data=df, **signature)  #x="earliness_median", y="pat_precision", hue="task", style='model', size='dataset_eval')
    if title:
        plt.title(title)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    legend.get_title().set_fontsize('15') 
    fig.tight_layout()  
    plt.savefig(fname, dpi=300)
 
def process_run(eval_file, pred_file):
    with open(eval_file, 'r') as f:
        try:
            d_ev = json.load(f)
        except: 
            print(eval_file)
    df = pd.DataFrame(d_ev)
    with open(pred_file, 'r') as f:
        d_p = json.load(f)
    
    physionet_vars = False 
    if 'dataset_kwargs' in d_p.keys(): 
        if 'only_physionet_features' in d_p['dataset_kwargs'].keys():
            if d_p['dataset_kwargs']['only_physionet_features']:
                physionet_vars = True
    if 'variable_set' in d_p.keys(): #sklearn pipe
        if d_p['variable_set'] == 'physionet':
            physionet_vars = True
    df['physionet_variables'] = physionet_vars
     
    #interesting keys from pred file:
    keys = ['model', 'dataset_train', 'dataset_eval', 'split', 'task', 'label_propagation', 'rep']
    pred_dict = {key: d_p[key] for key in keys if key in d_p.keys() }
    maybe_keys = ['subsample', 'subsampled_prevalence']
    for key in maybe_keys:
        if key in d_p.keys():
            pred_dict[key] = d_p[key]

    for key in pred_dict.keys():
        df[key] = pred_dict[key]
    if 'task' not in d_p.keys():
        raise ValueError('Task information is not available!')
    #try: result.update(pred_dict)
    #except: #task not available in deep jsons
    #    print(pred_file)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        required=True,
        type=str,
        help='Path to evaluation json files',
    )
    parser.add_argument(
        '--output_path',
        required=True,
        type=str,
        help='Path to where data for plotting is dumped, e.g `results/evaluation/plots`'
    )

    args = parser.parse_args()
    
    methods = ['lgbm', 'lr', 'sofa', 'qsofa', 'sirs', 'mews', 'news'] #attn model not in file name
    attn_datasets = ['EICU', 'AUMC', 'Hirid', 'MIMIC', 'Physionet2019'] 
    #deep model file names have this dataset namings (from datset cls)
    keys = methods + attn_datasets
 
    path = args.input_path #'results/evaluation/evaluation_output'
    output_path = args.output_path
 
    eval_files = [os.path.join(p, f) for (p, _, files) in os.walk(path) for f in files]
    pred_files = [f.replace('evaluation_output', 'prediction_output') for f in eval_files] 
    df_list = []
    used_eval_files = []
    used_pred_files = [] 
    for i, (eval_f, pred_f) in enumerate(zip(eval_files, pred_files)):
        if any([x in eval_f for x in keys]):
            used_eval_files.append(eval_f)
            used_pred_files.append(pred_f)
    df_list = Parallel(n_jobs=80, batch_size=50)(delayed(process_run)(eval_f, pred_f) for eval_f, pred_f in zip(used_eval_files, used_pred_files))
    for i, eval_f in enumerate(used_eval_files):
        df_list[i]['fname'] = eval_f 
        df_list[i]['job_id'] = i
    #curr_df = process_run(eval_f, pred_f)
    #curr_df['fname'] = eval_f
    #curr_df['job_id'] = i
    #df_list.append(curr_df)
    df = pd.concat(df_list, ignore_index=True)
    embed()
 
    for col in ['dataset_train', 'dataset_eval']:
        df[col] = df[col].apply(format_dataset)
    
    fname = 'result_data_subsampled.csv' if 'subsampled' in path else 'result_data.csv' 
    df.to_csv(os.path.join(output_path, fname), index=False)
  
