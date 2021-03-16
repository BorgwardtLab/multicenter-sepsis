import os
import numpy as np
from time import time
from src.sklearn.data.utils import load_pickle, save_pickle


def apply_label_shift(labels, shift):
    """Apply label shift to labels."""
    labels = labels.copy()
    patients = labels.index.get_level_values('id').unique()
    # sanity check: assert that no reordering occured:
    assert np.all(labels.index.levels[0] == patients)
    new_labels = pd.DataFrame() 
    
    for patient in patients:
        #labels[patient] = shift_onset_label(patient, labels[patient], shift)
        # the above (nice) solution lead to pandas bug in newer version.. 
        shifted_labels = shift_onset_label(patient, labels[patient], shift)
        df = shifted_labels.to_frame()
        df = df.rename(columns={0: 'sep3'}) 
        df['id'] = patient
        new_labels = new_labels.append(df)
    new_labels.reset_index(inplace=True)
    new_labels.set_index(['id', 'time'], inplace=True)
    return new_labels['sep3']

def handle_label_shift(args, d):
    """Handle label shift given argparse args and data dict d"""
    if args.label_propagation != 0:
        # Label shift is normally assumed to be in the direction of the future.
        # For label propagation we should thus take the negative of the
        # provided label propagation parameter
        cached_path = os.path.join('datasets', args.dataset, 'data', 'cached')
        cached_file = os.path.join(cached_path, f'y_shifted_{args.label_propagation}'+'_{}.pkl')
        cached_train = cached_file.format('train')
        cached_validation = cached_file.format('validation')
        cached_test = cached_file.format('test')
 
        if os.path.exists(cached_train) and not args.overwrite:
            # read label-shifted data from json:
            print(f'Loading cached labels shifted by {args.label_propagation} hours')
            y_train = load_pickle(cached_train)
            y_val = load_pickle(cached_validation)
            y_test = load_pickle(cached_test)
        else:
            # unpack dict
            y_train = d['y_train']
            y_val = d['y_validation'] 
            y_test = d['y_test']

            # do label-shifting here: 
            start = time()
            y_train = apply_label_shift(y_train, -args.label_propagation)
            y_val = apply_label_shift(y_val, -args.label_propagation)
            y_test = apply_label_shift(y_test, -args.label_propagation)

            elapsed = time() - start
            print(f'Label shift took {elapsed:.2f} seconds')
            #and cache data to quickly reuse from now:
            print('Caching shifted labels..')
            save_pickle(y_train, cached_train) #save pickle also creates folder if needed 
            save_pickle(y_val, cached_validation)
            save_pickle(y_test, cached_test)
        # update the shifted labels in data dict:
        d['y_train'] = y_train
        d['y_validation'] = y_val
        d['y_test'] = y_test 
    else:
        print('No label shift applied.')
    return d

