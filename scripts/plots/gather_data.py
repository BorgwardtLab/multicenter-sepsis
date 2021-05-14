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

def make_scatter(df, fname, signature, title=None):
    fig = plt.figure()
    sns.scatterplot(data=df, **signature)  #x="earliness_median", y="pat_precision", hue="task", style='model', size='dataset_eval')
    if title:
        plt.title(title)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    legend.get_title().set_fontsize('15') 
    fig.tight_layout()  
    plt.savefig(fname, dpi=300)
 
def process_run(eval_file, pred_file, recall_threshold=0.90):
    with open(eval_file, 'r') as f:
        try:
            d_ev = json.load(f)
        except: 
            print(eval_file)
    df = pd.DataFrame(d_ev)
    with open(pred_file, 'r') as f:
        d_p = json.load(f)
    
    greater = df['pat_recall'] > recall_threshold
    index = df['pat_recall'][greater].argmin()
    info = df.loc[index]
    result = info.to_dict()
    #interesting keys from pred file:
    keys = ['model', 'dataset_train', 'dataset_eval', 'split', 'task', 'label_propagation', 'rep']
    pred_dict = {key: d_p[key] for key in keys if key in d_p.keys() }
    maybe_keys = ['subsample', 'subsampled_prevalence']
    for key in maybe_keys:
        if key in d_p.keys():
            pred_dict[key] = d_p[key]

    df = pd.DataFrame(d_ev)
    for key in pred_dict.keys():
        df[key] = pred_dict[key]
    if 'task' not in d_p.keys():
        raise ValueError('Task information is not available!')
    try: result.update(pred_dict)
    except: #task not available in deep jsons
        print(pred_file)
    #TODO: currently result dict not used 
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        required=True,
        type=str,
        help='Path to evaluation json files',
    )
    args = parser.parse_args()

    methods = ['lgbm', 'sofa', 'qsofa', 'sirs', 'mews', 'news'] #attn model not in file name
    attn_datasets = ['EICU', 'AUMC', 'Hirid', 'MIMIC'] 
    #deep model file names have this dataset namings (from datset cls)
    keys = methods + attn_datasets
 
    path = args.input_path #'results/evaluation/evaluation_output'
    
    eval_files = [os.path.join(p, f) for (p, _, files) in os.walk(path) for f in files]
    pred_files = [f.replace('evaluation_output', 'prediction_output') for f in eval_files] 
    df_list = [] 
    for i, (eval_f, pred_f) in enumerate(zip(eval_files, pred_files)):
        if any([x in eval_f for x in keys]):
            curr_df = process_run(eval_f, pred_f)
            curr_df['fname'] = eval_f
            curr_df['job_id'] = i
            df_list.append(curr_df)
    df = pd.concat(df_list, ignore_index=True)
    
    for col in ['dataset_train', 'dataset_eval']:
        df[col] = df[col].apply(format_dataset)
    outpath ='results/evaluation/plots'
    fname = 'result_data_subsampled.csv' if 'subsampled' in path else 'result_data.csv' 
    df.to_csv(os.path.join(outpath, fname), index=False)
  
