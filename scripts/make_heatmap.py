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
         'physionet2019': 'Emory'
    }
    return d[name] 

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

    args = parser.parse_args()
    path = args.INPUT
    df = pd.read_csv(path)
    df = df.query('model == @args.model')
    
    for col in ['train_dataset','eval_dataset']: 
        df[col] = df[col].apply(dataset_naming)
    
    if not args.emory:
        drop = 'Emory'
        df = df.query('train_dataset != @drop & eval_dataset != @drop') 
 
    df = df.pivot(
        'train_dataset',
        columns='eval_dataset',
        values=args.metric
    )
    

    if args.no_diagonal:
        for dataset in df.columns:
            df[dataset][dataset] = np.nan

    sns.heatmap(df, cmap='Blues', annot=True, vmin=0.5, vmax=1.0)

    plt.tick_params(
        axis='both',
        which='major',
        labelleft=True,
        labelright=True,
        labelbottom=True,
        labeltop=True
    )
    outfile = os.path.split(path)[0]
    outfile = os.path.join(outfile, f'heatmap_{args.metric}_{args.model}')
    if args.emory:
        outfile += '_w_emory'
    if 'subsampled' in os.path.split(path)[-1]:
        outfile += '_subsampled'
    plt.savefig(outfile + '.png', dpi=400)
        

