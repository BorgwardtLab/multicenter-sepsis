"""Create heatmap from CSV of results."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import itertools
import sys
import os
from src.evaluation.patient_evaluation import format_dataset

def dataset_naming(name):
    name = format_dataset(name)
    d = {'mimic': 'MIMIC',
         'eicu': 'eICU',
         'hirid': 'HiRID',
         'aumc':  'AUMC',
         'physionet2019': 'Emory',
         'pooled': r'pooled $(n-1)$',
    }
    return d[name] 

renaming = {
    'train_dataset': 'training dataset',
    'eval_dataset': 'evaluation dataset'
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument(
        '-m', '--metric',
        type=str,
        default='auc_mean',
        help='Metric to use for edge weights'
    )
    parser.add_argument(
        '-d', '--no-diagonal',
        action='store_true',
        help='If set, removes diagonal from heat map'
    )
    parser.add_argument(
        '-M', '--model',
        default='AttentionModel',
        help='Select model to visualise'
    )
    parser.add_argument(
        '--emory',
        action='store_true',
        help='Add emory (slightly different setup)'
    )
    parser.add_argument(
        '--pooled_path',
        help='to add additional row of pooled predictions'
    )
    parser.add_argument('--digits', type=int, default=2, help='Number of digits after decimal.')

    args = parser.parse_args()
    path = args.INPUT
    df = pd.read_csv(path)
    df = df.query('model == @args.model')

    pooled_path = args.pooled_path
    if pooled_path is not None:
        pooled = pd.read_csv(pooled_path)
        pooled = pooled.query('model == @args.model')
        df = pd.concat([df,pooled])

    for col in ['train_dataset','eval_dataset']: 
        df[col] = df[col].apply(dataset_naming)
    
    if not args.emory:
        drop = 'Emory'
        df = df.query('train_dataset != @drop & eval_dataset != @drop') 

    # print mean +- std of AUCs, internal and external (if available)
    # internal auc:
    aucs = {}
    aucs['i_auc'] = df.query('train_dataset == eval_dataset')['auc_mean']
    # external auc:
    aucs['e_auc'] = df.query("train_dataset != eval_dataset & train_dataset != 'pooled $(n-1)$'")['auc_mean']
    if pooled_path is not None:
        # external with max pooling:
        aucs['ep_auc'] = df.query("train_dataset == 'pooled $(n-1)$'")['auc_mean'] 
    for key,val in aucs.items():
        mu = val.mean()
        sig = val.std()
        print(rf'{key}: $ {mu:.3f} \pm {sig:.3f}$ ') 
            
    df = df.pivot(
        'train_dataset',
        columns='eval_dataset',
        values=args.metric
    )
    df.columns.name = renaming[df.columns.name]
    df.index.name = renaming[df.index.name]

    if args.no_diagonal:
        for dataset in df.columns:
            df[dataset][dataset] = np.nan

    annotation_format = '.0{:d}f'.format(args.digits)
    g = sns.heatmap(df, fmt=annotation_format, cmap='Blues', annot=True, vmin=0.6, vmax=1.0, linewidth=2, linecolor='w')

    #g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')

    plt.tick_params(
        axis='both',
        which='major',
        labelleft=True,
        #labelright=True,
        #labelbottom=True,
        labeltop=True
    )
    g.figure.tight_layout()

    outfile = os.path.split(path)[0]
    outfile = os.path.join(outfile, f'heatmap_{args.metric}_{args.model}')
    if args.emory:
        outfile += '_w_emory'
    if 'subsampled' in os.path.split(path)[-1]:
        outfile += '_subsampled'
    plt.savefig(outfile + '.png', dpi=400)
        

