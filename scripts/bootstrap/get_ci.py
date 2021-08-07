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


def ci_interval(data, p=95):
    """ return percentile-based CI for percentage p (default 95%-CI)"""
    lower = (100 - p) / 2
    upper  = 100 - lower 
    ci_upper = np.percentile(data, upper) 
    ci_lower = np.percentile(data, lower)
    return ci_lower, ci_upper 

def gaussian_ci(data):
    """ return 95% CI assuming normality, i.e. +-1.96 SD"""
    mu = data.mean()
    sigma = data.std()
    lower = mu - 1.96*sigma
    upper = mu + 1.96*sigma
    return lower, upper

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

    df = pd.read_csv(input_path)
    target_cols = ['AUC'] if 'roc' in input_path else ['x','y']
    
    output = pd.DataFrame()
 
    for (model, train_dataset, eval_dataset), df_ in df.groupby(['model', 'train_dataset', 'eval_dataset']):
        ci_dict = {} 
        for col in target_cols:
            
            # current data for computing CI:
            CIs = ci_interval( df_[col] )
            ci_dict[col + '_lower'] = [CIs[0]]
            ci_dict[col + '_upper'] = [CIs[1]]
        ci_df = pd.DataFrame(ci_dict)
        # update row with further info
        ci_df['model'] = model
        ci_df['train_dataset'] = train_dataset
        ci_df['eval_dataset'] = eval_dataset
         
        output = output.append(ci_df)
    output_path = input_path.rstrip('.csv')
    output_path = output_path + '_confidence_intervals.csv' 
    output.to_csv(
        output_path,
        index=False
    )

    # Get CIs of mean over datasets!
    if 'pooled' not in input_path:
        # Internal:
        df = df.query('train_dataset == eval_dataset & train_dataset != "emory"')
    
    # First compute means over dataset for each rep and subsample 
    big_df = pd.DataFrame() #mean measures over all datasets

    rep_name = 'repetition' if 'scatter' in input_path else 'rep' 
    for (model, rep, subsample), df_ in df.groupby(['model', rep_name, 'subsample']):
        # gather means over datasets:
        means_dict = {}
        for col in target_cols:
            means_dict[col] = [df_[col].mean()]
        means_df = pd.DataFrame(means_dict)
        means_df['rep'] = rep  
        means_df['subsample'] = subsample 
        means_df['model'] = model
        big_df = big_df.append(means_df)

    # Next, compute CIs for those dataset means:
    mean_ci = pd.DataFrame()
    for (model), df_ in big_df.groupby(['model']):
        ci_dict = {} 
        for col in target_cols:
            # current data for computing CI:
            CIs = ci_interval( df_[col] )
            print(f'Model: {model}')
            print(f'Percentile intervals: {CIs}')
            GIs = gaussian_ci( df_[col] )
            print(f'Gaussian intervals: {GIs}') 
            ci_dict[col + '_lower'] = [CIs[0]]
            ci_dict[col + '_upper'] = [CIs[1]]
        ci_df = pd.DataFrame(ci_dict)
        # update row with further info
        ci_df['model'] = model
        mean_ci = mean_ci.append(ci_df)
    # Also write mean CIs to csv:
    output_path = input_path.rstrip('.csv')
    output_path = output_path + '_mean_confidence_intervals.csv' 
    mean_ci.to_csv(
        output_path,
        index=False
    )


if __name__ == '__main__':
    main()
