# script to load raw result data and check how sample rejections impact results:

import os
import pandas as pd
from pathlib import Path
import numpy as np
import json
import argparse
from IPython import embed

def read_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def write_json(obj, fname):
    with open(fname, 'w') as f:
        json.dump(obj, f, indent=4)

def main(args):
    d = read_json(args.input)
    scores = d['scores']    
    maxima = [np.max(s) for s in scores]

    # create mask (based on score percentiles)
    mask1 = maxima > np.percentile(maxima,args.lower) 
    mask2 = maxima < np.percentile(maxima,args.upper) 
    mask = mask1 * mask2 # percentile 40-60 
    
    # apply mask to all list-based keys:
    keys = []
    for key in d.keys():
        if type(d[key]) == list:
            keys.append(key)
    assert len(np.unique([len(d[key]) for key in keys])) == 1 # check that all keys have same length in the values 
    for key in keys:
        x = np.array(d[key]) #array of lists
        x = x[~mask] #apply mask (mask are ids we wish to drop)
        # remove int64 from numpy for json: 
        if type(x[0]) == np.int64:
            x = [int(i) for i in x]
        else: # otherwise just convert np arrays to list for json
            x = list(x)
        
        d[key] = x #overwrite  
     
    d['rejection_percentiles'] = [args.lower, args.upper]

    # write the processed file out to output_folder:
    output_file = os.path.join(
        args.output_dir,
        Path(*Path(args.input).parts[1:]) 
    )
    os.makedirs(os.path.split(output_file)[0],
        exist_ok=True)

    write_json(d, output_file) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default = 'results/evaluation_test/prediction_output_subsampled/j76ft4wm_EICU_subsample_0.json', 
        help = 'input path to raw result file'
    )
    parser.add_argument(
        '--output_dir',
        default = 'results/sample_rejections', 
        help = 'path to output folder'
    )
    parser.add_argument(
        '--lower',
        default = 40, type=int, 
        help = 'lower end of percentile for uncertainty / rejection mask'
    )
    parser.add_argument(
        '--upper',
        default = 60, type=int, 
        help = 'upper end of percentile for uncertainty / rejection mask'
    )

    args = parser.parse_args()
    main(args)


