import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from IPython import embed

input_path = '/links/groups/borgwardt/Projects/sepsis/multicenter-sepsis/results/baselines'
methods = ['sofa', 'sirs', 'qsofa', 'mews', 'news']
datasets = ['eicu', 'hirid', 'mimic3', 'aumc']

df = pd.DataFrame()
for dataset in datasets:
    for method in methods:
        fpath = os.path.join(input_path, '_'.join([dataset, method]), 'results.json')
        with open(fpath, 'r') as f:
            d = json.load(f)
        record = {}
        record['thres'] = [d['best_params']['est__theta'] ]
        keys = ['val_physionet_utility', 'val_roc_auc', 'val_average_precision', 
            'val_balanced_accuracy', 'method']
        for key in keys:
            record[key] = [d[key]]
        record['dataset'] = [dataset]
        # 'val_physionet_utility', 'val_roc_auc', 'val_average_precision', 
        # 'val_balanced_accuracy', 'method', 'best_params', 'n_iter_search', 'runtime', 'val_predict'
        record_df = pd.DataFrame(record)
        df = df.append(record_df)

#plot result:
cols = ['thres',
 'val_physionet_utility',
 'val_roc_auc',
 'val_average_precision',
 'val_balanced_accuracy',
 ]
#todo: pivot!

df = df.melt(id_vars=['method','dataset'])
plt.figure(figsize=(5,17))
g = sns.catplot(x='dataset', y='value', hue='method', row='variable', data=df, kind='bar', height=2, aspect=2)
#plt.legend()
plt.savefig(os.path.join(input_path,'baselines.png'), dpi=300)


        

