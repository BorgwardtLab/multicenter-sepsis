import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from IPython import embed
import argparse

def main(args):

    #base_path = os.path.join('results', 'pos_weight_analysis')
    #file_path = os.path.join(base_path, 'random_baselines_parallel_dictionary_lambda.csv') #'random_baselines_parallel_dictionary_u_fp_scorer.csv' 'random_baselines_parallel_dictionary_u_fp.csv' random_baselines_parallel_quadratic_u_fp.csv
    #df = pd.read_csv(file_path)
    
    df = pd.read_csv(args.input_file)

    plt.figure()
    g = sns.lineplot(x='p', y='utility', hue='dataset', data=df, err_style='bars')
    plt.title(f'Random baseline utility score performance')
    #plt.ylim((-2.5,1))
    plt.savefig(
        args.output_file,
        dpi=300
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file',
        help='input csv file containing random baseline performances', 
        required=True
    )
    parser.add_argument(
        '--output-file',
        help='filename (png) of output plot',
        required=True
    )
    args = parser.parse_args()
    main(args)
