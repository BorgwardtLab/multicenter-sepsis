import os
import numpy as np
from time import time
import pandas as pd
import pathlib

from src.evaluation import shift_onset_label

from src.sklearn.data.utils import load_pickle, save_pickle
from src.variables.mapping import VariableMapping

VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)

def apply_label_shift(labels, shift):
    """Apply label shift to labels."""
    labels = labels.copy()
    patients = labels.index.get_level_values(VM_DEFAULT('id')).unique()
    # added above sort as get_level_values did change order!
    # sanity check: assert that no reordering occured:
    assert np.all( labels.index.get_level_values(VM_DEFAULT('id')).unique() == patients)
    new_labels = pd.DataFrame() 
    
    for patient in patients:
        #labels[patient] = shift_onset_label(patient, labels[patient], shift)
        # the above (nice) solution lead to pandas bug in newer version.. 
        shifted_labels = shift_onset_label(patient, labels[patient], shift)
        df = shifted_labels.to_frame()
        df = df.rename(columns={0: VM_DEFAULT('label')}) 
        df[VM_DEFAULT('id')] = patient
        new_labels = new_labels.append(df)
    new_labels.reset_index(inplace=True)
    new_labels.set_index([VM_DEFAULT('id'), VM_DEFAULT('time')], inplace=True)
    return new_labels[VM_DEFAULT('label')]

def handle_label_shift(args, d):
    """Handle label shift given argparse args and data dict d"""
    if args.label_propagation != 0:
        ## Label shift is normally assumed to be in the direction of the future.
        ## For label propagation we should thus take the negative of the
        ## provided label propagation parameter
        #cached_path = os.path.join('datasets', args.dataset, 'data', 'cached')
        #cached_file = os.path.join(cached_path, f'y_shifted_{args.label_propagation}'+'_{}.pkl')
        #cached_train = cached_file.format('train')
        #cached_validation = cached_file.format('validation')
        #cached_test = cached_file.format('test')
 
        #if os.path.exists(cached_train) and not args.overwrite:
        #    # read label-shifted data from json:
        #    print(f'Loading cached labels shifted by {args.label_propagation} hours')
        #    y_train = load_pickle(cached_train)
        #    y_val = load_pickle(cached_validation)
        #    y_test = load_pickle(cached_test)
        #else:
        # unpack dict
        # keys to shift:
        keys = ['y_train','y_validation','y_test', 'tp_labels_shifted_validation']
        start = time()
        for key in keys:
            if key in d.keys():
                print(f'Shifting {key} ..')
                d[key] = apply_label_shift(d[key], -args.label_propagation)
        elapsed = time() - start
        print(f'Label shift took {elapsed:.2f} seconds')
        #and cache data to quickly reuse from now:
    else:
        print('No label shift applied.')
    return d

