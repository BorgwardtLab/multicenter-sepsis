"""Plot every datum in an evaluation file (work in progress).

Example call:

    python -m scripts.plots.plot_panacea      \
        --input_path evaluations.json         \
        --predictions_path predictions.json   \
        --output_path /tmp/

This will create a plot in `tmp`.
"""

import argparse
import itertools
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from src.evaluation.patient_evaluation import format_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='results/evaluation/patient_eval_lgbm_aumc_aumc.json')
    parser.add_argument('--predictions_path', required=True)
    parser.add_argument('--output_path', default='results/evaluation/plots')
    parser.add_argument('--earliness-stat', default='mean')
    parser.add_argument('--level', default='pat', help='pat or tp level')
    parser.add_argument('--recall_thres', default=0.9, type=float, help='recall threshold [0-1]')

    parser.add_argument('-s', '--show', action='store_true')

    args = parser.parse_args()
    input_path = args.input_path
    with open(input_path, 'r') as f:
        d = json.load(f)
    df = pd.DataFrame(d)
    level = args.level
    recall_thres = args.recall_thres

    with open(args.predictions_path, 'r') as f:
        d = json.load(f)
        scores = list(itertools.chain.from_iterable(d['scores']))
        names = {}
        names['model'] = d['model']
        names['train_dataset'] = d['dataset_train']
        names['eval_dataset'] = d['dataset_eval']
        names['task'] = d['task'] 

    earliness_stat = args.earliness_stat
    earliness = f'earliness_{earliness_stat}'

    df_test = (df - df.min()) / (df.max() - df.min())
    df_test['label'] = ['x'] * len(df)

    pd.plotting.parallel_coordinates(
        df_test.drop(
            axis='columns',
            labels=[col for col in df.columns if 'proportion' in col]
        ),
        'label',
        color='black',
        alpha=0.5,
    )

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.gca().legend().set_visible(False)

    if args.show:
        plt.show()
