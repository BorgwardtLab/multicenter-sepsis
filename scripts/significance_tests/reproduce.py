import scipy.stats as stats 
import numpy as np
import pandas as pd
from IPython import embed
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # hard coded result files:
    f1 = 'results/evaluation_test/plots/roc_pooled_bootstrap_subsampled.csv'
    f2 = 'results/evaluation_test/plots/roc_bootstrap_subsampled.csv'

    f1 = '/Users/mimoor/Desktop/localwork/projects/mcsepsis/multicenter-sepsis/revisions/results/evaluation_test/prediction_pooled_subsampled/max/plots/roc_bootstrap_subsampled.csv'
    f2 = '/Users/mimoor/Desktop/localwork/projects/mcsepsis/multicenter-sepsis/revisions/results/evaluation_test/plots/roc_bootstrap_subsampled.csv'

    dfs = [pd.read_csv(f) for f in (f1, f2)]
    
    # check to remove redundant index column
    drop_col = 'Unnamed: 0'
    dfs_new = []
    for df in dfs:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
        dfs_new.append(df)
        print(df.columns)
    dfs = dfs_new

    # join pooled and single dataset results
    df = pd.concat(dfs)
    datasets = ['aumc', 'eicu', 'hirid', 'mimic']

    # emory is reported seperately:
    df = df.query("train_dataset != 'emory'  & eval_dataset != 'emory' & model == 'AttentionModel'")
    
    # for each evaluation site (core datasets),
    # get paired results: pooled vs mean of single dataset perf.
    # firsts, get pooled results:
    df_p = df.query("train_dataset == 'pooled'")
    # get single dataset results:
    df_s = df.query("train_dataset != 'pooled' & train_dataset != eval_dataset")
    

    #first get means of single dataset performances (expected performance on single dataset)
    means = []
    for dataset, df_ in df_s.groupby('eval_dataset'):
        curr_means = df_.groupby(['rep','subsample']).max() #mean()
        for _, rep_df in df_.groupby(['rep','subsample']):
            ind = rep_df['AUC'].argmax()
            best = rep_df.iloc[[ind]]
            means.append(best)
        #curr_means['eval_dataset'] = dataset
        #means.append(curr_means)
    df_means = pd.concat(means)
    
    # iterate over datasets and run the test:
    for dataset in datasets:
        mu = df_means.query("eval_dataset == @dataset")
        pooled = df_p.query("eval_dataset == @dataset")
        #this step is merely to sort and arrange the comparison array similarly 
        pooled = pooled.groupby(['rep','subsample']).max() 
        
        print('>>> ',dataset)
        print('Shapiro-Wilk test for normality:')
        print('Pooled AUCs:', stats.shapiro(pooled['AUC']))
        print('Mean Single AUCs:', stats.shapiro(mu['AUC']))
        print('Paired t-test:')
        print(stats.ttest_rel(mu['AUC'], pooled['AUC']))
        print('Wilcoxon signed rank test:')
        wc = stats.wilcoxon(mu['AUC'], pooled['AUC'])
        print(stats.wilcoxon(mu['AUC'], pooled['AUC']))

        plt.figure()
        plt.title(f' {dataset}, AUC values: best single vs pooled, wilcoxon p = {wc.pvalue:.2E}')
        plt.hist(mu['AUC'], label='best single', alpha=0.5)
        plt.hist(pooled['AUC'], label='pooled', alpha=0.5)
        plt.legend()
        plt.ylabel('counts')
        plt.xlabel('AUC')
        plt.savefig(f'reproduce_pooled_vs_single_wc_test_{dataset}.png')

if __name__ == "__main__":
    main()
