import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from IPython import embed
import os
import pandas as pd
import numpy as np 

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', default='results/evaluation/extended_features/patient_eval_lgbm_aumc_aumc.json')
parser.add_argument('--output_path', default='results/evaluation/extended_features/')

args = parser.parse_args()
input_path = args.input_path
with open(input_path, 'r') as f:
    d = json.load(f)
df = pd.DataFrame(d)
df['earliness_mean'] *= -1 

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.set_title('Patient-based Evaluation', fontsize=19)
ax1.set_xlabel('Decision threshold', fontsize=16)
ax1.set_ylabel('Score', fontsize=16, color='green')
#ax1 = sns.lineplot(x='thres', data = df[['thres', 'pat_recall','pat_precision']])
ax1 = sns.lineplot(x='thres', y='pat_precision', data = df, label='precision', color='darkgreen')
ax1 = sns.lineplot(x='thres', y='pat_recall', data = df, label='recall', color='tab:green')

ax2 = ax1.twinx()
ax2.set_ylabel('Earliness: mean #hours before onset', fontsize=16, color='red')
ax2 = sns.lineplot(x='thres', y='earliness_mean', data = df, label='earliness', color='red')

plt.xticks(np.arange(0, 1, step=0.1))
ax1.set_yticks(np.arange(0,1, step=0.1))

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0.05,0.8))

out_file = os.path.split(input_path)[-1].split('.')[0] + '.png' 
plt.savefig( os.path.join(args.output_path, out_file))
