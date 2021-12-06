# Test whether low earliness in Emory is correct

import json
from IPython import embed
import numpy as np

def load_dict(f):
    with open(f,'r') as F:
        return json.load(F)
    
def get_alarms(d_pred, thres):
    alarms = {}
    for i, pat_scores in enumerate(d_pred['scores']):
        for j, score in enumerate(pat_scores):
            if score >= thres:
                # raise alarm for patient i, at time j
                assert i not in alarms.keys() # prevent overwriting
                alarms[i] = j
                break
    return alarms

def get_onsets(labels):
    """ labels: list (patients) of list (timestep labels) """
    onsets = {}
    for i, pat_labels in enumerate(labels):
        for j, label in enumerate(pat_labels):
            if label == 1: 
                assert i not in onsets.keys() # prevent overwriting
                onsets[i] = j
                break
    return onsets

def main():
    # example (eval and pred) file:
    eval_path = 'results/evaluation_test/evaluation_output_subsampled/5a65k0fo_Physionet2019_subsample_0.json' 
    pred_path = 'results/evaluation_test/prediction_output_subsampled/5a65k0fo_Physionet2019_subsample_0.json'
    thres = 0.0548 # plot_patient_eval outputs this threshold at 80% Sensitivity

    # Load eval measures and predictions 
    d_ev = load_dict(eval_path)
    d_pred = load_dict(pred_path) 
   
    # Loop over prediction scores and raise alarm when score >= thres
    alarms = get_alarms(d_pred, thres)
    onsets_a = get_onsets(d_pred['labels'])
    onsets_b = get_onsets(d_pred['targets'])

    # compute delta earliness distribution:
    deltas = {}
    for pat_ind in alarms.keys():
        if pat_ind in onsets_a.keys():
            deltas[pat_ind] = onsets_a[pat_ind] - alarms[pat_ind]
    
    earliness = np.median(list(deltas.values()))

    print(f'Earliness median = {earliness}')
 

if __name__ == '__main__':
    main()
