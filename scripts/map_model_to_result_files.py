"""This script reads all prediction jsons in a given input folder and creates a mapping from (model, dataset, subsample) to filenames """
import json
import glob
import argparse
from collections import defaultdict
from IPython import embed
from joblib import Parallel, delayed
from src.evaluation.patient_evaluation import format_dataset

def load_json(f):
    with open(f,'r') as F:
        return json.load(F)

def extract_keys(d):
    keys = [
        'model', 'split', 'rep', 'subsample', 'run_id'    
    ]
    data_keys = ['dataset_train', 'dataset_eval']

    avail_keys = [key for key in keys if key in d.keys()] 
    out = {key: d[key] for key in avail_keys} 
    data_d = {key: format_dataset(d[key]) for key in data_keys}
    out.update(data_d) 
    return out

def process_file(f):
    d = load_json(f)
    d = extract_keys(d)
    d['filename'] = f
    return d
    
def main(args):
    output_path = args.output_path
    if args.overwrite:
        files = glob.glob(args.INPUT + '*.json')

        nested_dict = lambda: defaultdict(nested_dict)
        fmap = nested_dict() #file mapping dict
          
        results = Parallel(n_jobs=50, batch_size=50)(delayed(process_file)(f) for f in files) #n_jobs=50
        subsampled = False
        if 'subsample' in args.INPUT:
            subsampled = True  
        for d in results:
            if args.finetuning and 'run_id' in d.keys():
                rep_or_run = 'run_'+str(d['run_id']) 
            else: 
                rep_or_run = 'rep_'+str(d['rep'])
            if subsampled:
                fmap[d['model']][d['dataset_train']][d['dataset_eval']][rep_or_run]['subsample_'+str(d['subsample'])] = d['filename']
            else:
                fmap[d['model']][d['dataset_train']][d['dataset_eval']][rep_or_run] = d['filename']

        with open(output_path, 'w') as F:
            json.dump(fmap, F, indent=4)
    else:
        print('Loading cached/precomputed prediction mapping json..')
        fmap = load_json(output_path)

    # harmonize dataset names in the dict:
     

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='input folder to create mapping')
    parser.add_argument('--overwrite', action='store_true', help='compute anew')
    parser.add_argument('--output_path', default='results/evaluation_test/prediction_mapping.json')
    parser.add_argument('--finetuning', action='store_true', help='use run_id instead of rep in mapping')

    args = parser.parse_args()

    main(args)

    
