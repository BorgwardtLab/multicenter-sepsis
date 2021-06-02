"""Visualize results of shap analysis."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# plt.style.use('dark_background')
input_file = 'shap_test.pickle'

with open(input_file, 'rb') as f:
    data = pickle.load(f)

data
feature_names = data['feature_names']
instance = 0
input_data = data['input'][instance].numpy()
length = data['lengths'][instance]
data.keys()
# We get shap values for each prediction output, for now only take those of the
# last one
shap_values = data['shap_values'][0][instance]

# For now ignore indicator and count features
keep_features = [True if not col.endswith('indicator') and not col.endswith('count') else False for col in feature_names]
important_indices = np.where(keep_features)[0]
selected_features = np.array(feature_names)[important_indices]

# The first 10 features are from the positional embedding
important_indices += 10   # ignore positional encoding

def get_interpolated_values(values, n_values=200):
    # TODO: This is not completely correct, we should probably save the exact
    # time in the shap evaluation script and use them here. For now this should
    # be enough.
    x_values = range(len(values))
    f = interp1d(x_values, values)
    x = np.linspace(0, len(values)-1, n_values)
    return f(x)

fig, axs = plt.subplots(nrows=len(selected_features) // 2, ncols=2, sharex=True, figsize=(2*5, 1*(len(selected_features) // 2)))
axs = np.ravel(axs)
for i, (name, ax) in enumerate(zip(selected_features, axs)):
    cur_values = input_data[:, important_indices[i]]
    cur_shap = shap_values[:, important_indices[i]]
    min_val, max_val = np.nanmin(cur_values)*0.9, np.nanmax(cur_values)*1.1
    interpolated_shap = get_interpolated_values(cur_shap)[None, :]
    # TODO: Use actual time values instead of range statement
    ax.imshow(interpolated_shap, aspect='auto', extent=(0, length-1, min_val, max_val))
    ax.plot(range(len(cur_values)), cur_values, label="values")
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel(name)

fig.show()
