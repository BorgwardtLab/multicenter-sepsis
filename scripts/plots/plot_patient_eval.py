import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from IPython import embed
import os
import pandas as pd
import numpy as np 

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', default='results/evaluation/patient_eval_lgbm_aumc_aumc.json')
parser.add_argument('--output_path', default='results/evaluation/plots')
parser.add_argument('--earliness-stat', default='mean')
args = parser.parse_args()
input_path = args.input_path
with open(input_path, 'r') as f:
    d = json.load(f)
df = pd.DataFrame(d)

earliness_stat = args.earliness_stat
earliness = f'earliness_{earliness_stat}'


# Get precision at 90% Recall
greater = df['pat_recall'] > 0.9
index = df['pat_recall'][greater].argmin()
info = df.loc[index]
print(info)
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.set_title('Patient-based Evaluation', fontsize=19)
ax1.set_xlabel('Decision threshold', fontsize=16)
ax1.set_ylabel('Score', fontsize=16, color='green')
#ax1 = sns.lineplot(x='thres', data = df[['thres', 'pat_recall','pat_precision']])
ax1 = sns.lineplot(x='thres', y='pat_precision', data = df, label='precision', color='darkgreen')
ax1 = sns.lineplot(x='thres', y='pat_recall', data = df, label='recall', color='lightgreen') #tab:green
ax1 = sns.lineplot(x='thres', y='physionet2019_utility', data = df, label='utility', color='black') 

ax2 = ax1.twinx()
ax2.set_ylabel(f'Earliness: {earliness_stat} #hours before onset', fontsize=16, color='red')
ax2 = sns.lineplot(x='thres', y=earliness, data = df, label='earliness', color='red')

plt.xticks(np.arange(df['thres'].min(), df['thres'].max(), step=0.05))
ax1.set_yticks(np.arange(0,1, step=0.1))

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0.05,0.85))
ax1.get_legend().remove()

out_file = os.path.split(input_path)[-1].split('.')[0] + '_' + earliness + '.png' 
plt.savefig( os.path.join(args.output_path, out_file))
