# script to load raw result data and check how sample rejections impact results:

import pandas as pd
import numpy as np
import json
import argparse
from IPython import embed

def read_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def main(args):
    d = read_json(args.input)
    scores = d['scores']    
    maxima = [np.max(s) for s in scores]

    # create mask (based on score percentiles)
    mask1 = maxima > np.percentile(maxima,40) 
    mask2 = maxima < np.percentile(maxima,60) 
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
        d[key] = x #overwrite  
    embed() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default = 'results/evaluation_test/prediction_output_subsampled/j76ft4wm_EICU_subsample_0.json', 
        help = 'input path to raw result file'
    ) 
    args = parser.parse_args()
    main(args)


