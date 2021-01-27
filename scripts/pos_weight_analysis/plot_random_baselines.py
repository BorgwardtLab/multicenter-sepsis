import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from IPython import embed

base_path = os.path.join('results', 'pos_weight_analysis')
file_path = os.path.join(base_path, 'random_baselines.csv')

df = pd.read_csv(file_path)

g = sns.lineplot(x='p', y='utility', hue='dataset', data=df, err_style='bars')
plt.title('Utility of a random predictor: Bernoulli distribution with parameter p')
plt.savefig(
    os.path.join(base_path, 'random_baselines.png'),
    dpi=300
)
