import glob
from IPython import embed
import os 
import json
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation.patient_evaluation import format_dataset 

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
        d_ev = json.load(f)
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
    if 'task' not in d_p.keys():
        raise ValueError('Task information is not available!')
        ##TODO: ADD TASK IN DEEP JSONS!!
        #if any([id in d_p['run_id'] for id in ['r68b6gm4', 's453x0n9' ]]):
        #    task  = 'regression'
        #else: 
        #    task = 'classification'
        #pred_dict['task'] = task 
    try: result.update(pred_dict)
    except: #task not available in deep jsons
        print(pred_file) 
    return result

if __name__ == "__main__":
    path = 'results/evaluation/evaluation_output'
    eval_files = [os.path.join(p, f) for (p, _, files) in os.walk(path) for f in files]
    pred_files = [f.replace('evaluation_output', 'prediction_output') for f in eval_files] 
    dicts = [] 
    for eval_f, pred_f in zip(eval_files, pred_files):
        if 'lgbm' in eval_f:
            dicts.append(process_run(eval_f, pred_f))
    df = pd.DataFrame(dicts)

    for col in ['dataset_train', 'dataset_eval']:
        df[col] = df[col].apply(format_dataset)
    outpath ='results/evaluation/plots'
    df.to_csv(os.path.join(outpath, 'scatter.csv'))
  
    #overall plot:
    fname = os.path.join(outpath, f'scatter_overall.png')
    #sns.scatterplot(data=df, x="earliness_median", y="pat_precision", hue="dataset_eval", style='model')
    signature = { 'x':"earliness_median", 'y':"pat_precision", 'hue': "dataset_eval", 'style': 'model'}
    make_scatter(df, fname, signature)
    
    # task plot:
    fname = os.path.join(outpath, f'scatter_task.png')
    #sns.scatterplot(data=df, x="earliness_median", y="pat_precision", hue="task", style='model', size='dataset_eval')
    signature = { 'x':"earliness_median", 'y':"pat_precision", 'hue': "task", 'style': 'model', 'size': 'dataset_eval'}
    title = f'Performance at Recall 90%, Task comparison'
    make_scatter(df, fname, signature, title)

    # internal val plot:
    fname = os.path.join(outpath, f'scatter_internal.png')
    df_curr = df[df['dataset_train'] == df['dataset_eval']] 
    signature = { 'x':"earliness_median", 'y':"pat_precision", 'hue': "dataset_train", 'style': 'model'}
    title = f'Performance at Recall 90%, Internal validation'
    make_scatter(df_curr, fname, signature, title)

    # external val plot:
    fname = os.path.join(outpath, f'scatter_external.png')
    df_curr = df[df['dataset_train'] != df['dataset_eval']] 
    signature = { 'x':"earliness_median", 'y':"pat_precision", 'hue': "dataset_train", 'style': 'dataset_eval'}
    title = f'Performance at Recall 90%, External validation'
    make_scatter(df_curr, fname, signature, title)

    # per dataset plot:
    for dataset in ['mimic','hirid','aumc']:
        fname = os.path.join(outpath, f'scatter_{dataset}.png')
        df_curr = df[df['dataset_train'] == dataset]
        signature = { 'x':"earliness_median", 'y':"pat_precision", 'hue': "dataset_eval", 'style': 'model', 'size': 'task'}
        title = f'Performance at Recall 90%, trained on {dataset}'
        make_scatter(df_curr, fname, signature, title)
        #fig = plt.figure()
        #sns.scatterplot(data=df_curr, x="earliness_median", y="pat_precision", hue="dataset_eval", style='model', size='task')
        #plt.title(f'Performance at Recall 90%, trained on {dataset}')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #fig.tight_layout()  
        #plt.savefig(fname, dpi=300)
