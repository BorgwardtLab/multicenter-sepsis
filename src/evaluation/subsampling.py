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

def check_prevalence(output):
    y = [np.any(x).astype(int) for x in output['labels']]
    prev = np.sum(y) / len(y)
    print(f"Subsample {output['subsample']}, prevalence = {prev:.5f}")
    return prev

def extract_case_ids(output, cases=[]):
    """extract ids of cases for sanity checking coverage"""
    y = [np.any(x) for x in output['labels']] 
    case_ids = np.array(output['ids'])[y].tolist() 
    for i in case_ids:
        if i not in cases:
            cases.append(i)
    return cases 

def check_coverage(case_ids, d):
    total_cases = extract_case_ids(d)
    # defensive step (ensure no duplicates are available/counted)
    cases = np.unique(case_ids).shape[0]
    total = np.unique(total_cases).shape[0]
    coverage = cases / total
    print(f'Coverage = {coverage*100:.1f} %') 


def main(args):
    input_path = args.input_file 
    info = load_json(args.subsampling_file)
    d = load_json(input_path)
    
    # format of info dict:   
    # validation subsampling ids:
    # info['dev']['split_0']['validation_subsamples']['rep_0']['ids']
    # test subsampling ids:
    # info['test']['total_subsamples']['rep_0']['ids']
    # if using test repetitions: info['test']['split_0_subsamples']['rep_0']['ids'] 
    
    # keys (with patient-level information) to subset:
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

    case_ids = []
    for i in range(n_subsamples):
        subsample_ids = subsamples[f'rep_{i}']['ids']
        mask = np.isin(d['ids'], subsample_ids) 
        output = d.copy()
        for key in keys:
            output.pop(key)
            as_array = np.array(d[key], dtype=object)
            output[key] = as_array[mask].tolist()

        # sanity check all labels are assigned correctly:
        for j, id in enumerate(subsample_ids):
            index = np.isin(output['ids'], id)
            index = np.argmax(index) 
            info_label = subsamples[f'rep_{i}']['labels'][j]
            d_label = output['labels'][index]
            d_case = np.any(d_label)
            if info_label != d_case:
                print(id)
                #embed() 
            #assert info_label == d_case

        output['subsample'] = i
        prev = check_prevalence(output)
        output['subsampled_prevalence'] = prev

        outfile = os.path.split(input_path)[-1].rstrip('.json')
        outfile += f'_subsample_{i}.json'
        outfile = os.path.join(args.output_dir, outfile)    
        with open(outfile, 'w') as f:
            json.dump(output, f, indent=4)
  
        case_ids = extract_case_ids(output, case_ids) 
    check_coverage(case_ids, d)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file',
        required=True,
        type=str,
        help='Path to JSON file for which to run the evaluation',
    )
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
