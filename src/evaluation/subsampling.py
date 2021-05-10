import json
import pandas as pd
import argparse
import os
from IPython import embed
import sys 
import numpy as np

def load_json(fpath):
    with open(fpath,'r') as f:
        return json.load(f)

def main(args):
    #root = '/links/groups/borgwardt/Projects/sepsis/multicenter-sepsis/results/evaluation/old_results/sweep6_reps/prediction_output'
    root = '/links/groups/borgwardt/Projects/sepsis/multicenter-sepsis/results/evaluation/prediction_output' 
    input_path = os.path.join(root, 'lgbm_aumc_aumc_classification_middle_cost_5_50_iter_rep_0.json')

    info = load_json(args.subsampling_file)
    d = load_json(input_path)

    # format of info dict:   
    # validation subsampling ids:
    # info['dev']['split_0']['validation_subsamples']['rep_0']['ids']
    # test subsampling ids:
    # info['test']['total_subsamples']['rep_0']['ids']
    # if using test repetitions: info['test']['split_0_subsamples']['rep_0']['ids'] 
    
    # keys to subset:
    keys = ['ids', 'labels', 'targets', 'predictions', 'scores', 'times'] 
    
    split = d['split']
    assert split in ['validation', 'test']
    fold = d['rep']
    print(f'Processing {split} split..')
    if split == 'validation':
        subsamples = info['dev'][f'split_{fold}']['validation_subsamples']
    elif split == 'test':
        subsamples = info['test']['total_subsamples']
    n_subsamples = len(subsamples.keys())

    for i in range(n_subsamples):
        subsample_ids = subsamples[f'rep_{i}']['ids']
        mask = np.isin(d['ids'], subsample_ids) 
        output = d.copy()
        for key in keys:
            output.pop(key)
            as_array = np.array(d[key], dtype=object)
            output[key] = as_array[mask].tolist()

        outfile = os.path.split(input_path)[-1].rstrip('.json')
        outfile += f'_subsample_{i}.json'
        outfile = os.path.join(args.output_dir, outfile)    
        with open(outfile, 'w') as f:
            json.dump(output, f)

        # sanity check all labels are correct:
        for j, id in enumerate(subsample_ids):
            index = np.isin(output['ids'], id)
            index = np.argmax(index) 
            info_label = subsamples[f'rep_{i}']['labels'][j]
            d_label = output['labels'][index]
            d_case = np.any(d_label)
            if not info_label == d_case:
                print(id) 
                embed()
        #y1 = subsamples[f'rep_{i}']['labels']
        #y2 = output['labels']  
        #embed() 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    '--input-file',
    #    required=True,
    #    type=str,
    #    help='Path to JSON file for which to run the evaluation',
    #)
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output dir path for storing JSON files with subsampled predictions '
    )
    parser.add_argument(
        '--subsampling-file',
        required=True,
        help='path to JSON file containing subsampling ids'
    )
    args = parser.parse_args()

    main(args)
