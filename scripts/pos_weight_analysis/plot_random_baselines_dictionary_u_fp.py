import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from IPython import embed

base_path = os.path.join('results', 'pos_weight_analysis')
file_path = os.path.join(base_path, 'random_baselines_parallel_dictionary_lambda.csv') #'random_baselines_parallel_dictionary_u_fp_scorer.csv' 'random_baselines_parallel_dictionary_u_fp.csv' random_baselines_parallel_quadratic_u_fp.csv

df = pd.read_csv(file_path)


plt.figure()
g = sns.lineplot(x='p', y='utility', hue='dataset', data=df, err_style='bars')
plt.title(f'Heuristically aligned Utility score')
#plt.ylim((-2.5,1))
plt.savefig(
    os.path.join(base_path, f'random_baselines_dictionary_lambda.png'), #random_baselines_dictionary_u_fp.png  random_baselines_quadratic_u_fp.png'
    dpi=300
)
