import json 
from IPython import embed
import os
import subprocess 
import numpy as np
from tqdm import tqdm
import pandas as pd

def update_dict_w_key(d, key):
    if key not in d.keys():
        d[key]  = {}
    return d 

def interpolate_at(df, x):
    """Interpolate a data frame at certain positions.

    This is an auxiliary function for interpolating an indexed data
    frame at a certain position or at certain positions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame; must have index that is compatible with `x`.

    x : scalar or iterable
        Index value(s) to interpolate the data frame at. Must be
        compatible with the data type of the index.

    Returns
    -------
    Data frame evaluated at the specified index positions.
    """
    # Check whether object support iteration. If yes, we can build
    # a sequence index; if not, we have to convert the object into
    # something iterable.
    try:
        _ = (a for a in x)
        new_index = pd.Index(x)
    except TypeError:
        new_index = pd.Index([x])

    # Ensures that the data frame is sorted correctly based on its
    # index. We use `mergesort` in order to ensure stability. This
    # set of options will be reused later on.
    sort_options = {
        'ascending': False,
        'kind': 'mergesort',
    }
    df = df.sort_index(**sort_options)

    # TODO: have to decide whether to keep first index reaching the
    # desired level or last. The last has the advantage that it's a
    # more 'pessimistic' estimate since it will correspond to lower
    # thresholds.
    df = df[~df.index.duplicated(keep='last')]

    # Include the new index, sort again and then finally interpolate the
    # values.
    df = df.reindex(df.index.append(new_index).unique())
    df = df.sort_index(**sort_options)

    df = df.interpolate()
    res = df.loc[new_index]
    
    # recover NaN cols (strings that were not interpolated)
    first_row = df.iloc[0] # recover string cols from here
    assert df.isnull().iloc[0].sum() == 0
    recover_cols = ['model', 'dataset_train', 'dataset_eval', 'split', 'task', 'fname']
    for col in recover_cols:
        res[col] = first_row[col]
    return res

if __name__ == "__main__": 

    output_path = 'internal_preds_for_drago_temporal_analysis' #revisions/results/evaluation_test/

    with open('prediction_subsampled_mapping.json', 'r') as f:
        d = json.load(f)

    df = pd.read_csv('plots/result_data_subsampled.csv')
    query = 'model=="AttentionModel" and dataset_train==dataset_eval'
    df1 = df.query(query)
    df1 = df1.set_index('pat_recall')
    seen_files = []
    result = pd.DataFrame()
    for _, df_ in df1.groupby(['dataset_train', 'rep', 'subsample']):
        fname = df_['fname'].unique()
        assert len(fname) == 1 #only 1 source file used per df_
        fname = fname[0]
        assert fname not in seen_files
        seen_files.append(fname)
        res = interpolate_at(df_, 0.8)
        res.index = res.index.set_names(['pat_recall']) # index name
        res = res.reset_index()
        result = result.append(res)
    embed()
    result.to_csv(output_path + '/evaluation_results_at_80recall.csv')

    sys.exit()


    data = d['AttentionModel']
    # dict with keys [hirid, mimic, ..] (train dataset) 
    # each being a dict with keys [hirid, mimic,..] (test dataset)

    d_internal = {} # all internal predictions

    # Collect all files to copy to output_path
    for d_train in data.keys():
        for d_test in data[d_train].keys():
            if d_train != d_test:
                continue
            curr = data[d_train][d_test]
            d_internal[d_train] = curr

    # Copy prediction files:
    ## output = os.path.join(output_path, 'prediction_output_subsampled')
    output = os.path.join(output_path, 'evaluation_output_subsampled')
    eval_mode = True if 'evaluation' in output else False
    print(f'Eval mode = {eval_mode}')

    # mapping between source and destination paths:
    moving_dict = {}
    # structure of d_internal: d_internal['hirid']['rep_0-4']['subsample_0-9']
    for dataset in d_internal.keys():
        dataset_dict = {}
        for rep in d_internal[dataset].keys():
            dataset_dict = update_dict_w_key(dataset_dict, rep)
            
            for subsample in d_internal[dataset][rep].keys():
                dataset_dict[rep] = update_dict_w_key(dataset_dict[rep], subsample)
                
                source_path = d_internal[dataset][rep][subsample]
                source_path = '/'.join(source_path.split('/')[3:]) # make it relative 
                destination_path = os.path.join(
                        output,
                        f'{dataset}_{rep}_' + os.path.split(source_path)[-1]
                )
                if eval_mode:
                    source_path = source_path.replace('prediction','evaluation')
                print(source_path)
                print(f'{destination_path}')
                d_curr = dataset_dict[rep][subsample] 

                d_curr['source'] = source_path
                d_curr['destination'] = destination_path
                dataset_dict[rep][subsample] = d_curr
        moving_dict[dataset] = dataset_dict

    # Validate that the moving dict is not faulty:
    s_paths = []
    d_paths = []
    for dataset in moving_dict.keys():
        for rep in moving_dict[dataset].keys():
            for subsample in moving_dict[dataset][rep].keys():
                curr = moving_dict[dataset][rep][subsample]
                s_paths.append(curr['source'])
                d_paths.append(curr['destination'])
    print(len(s_paths))
    print(f'Source paths Unique len: {np.unique(s_paths).shape}')
    print(len(d_paths))
    print(f'Destination paths Unique len: {np.unique(d_paths).shape}')

    os.makedirs(output,exist_ok=True)
    # Actually copying the files:
    total = 4*5*10
    with tqdm(total=total) as pbar:
        for dataset in moving_dict.keys():
            for rep in moving_dict[dataset].keys():
                for subsample in moving_dict[dataset][rep].keys():
                    curr = moving_dict[dataset][rep][subsample]
                    source = curr['source']
                    dest = curr['destination']
                    if os.path.exists(dest):
                        print(f'Output file {dest} already exists, skipping!')
                        pbar.update(1)
                        continue
                    command = ['cp', source, dest]
                    subprocess.run(command, check=True)
                    pbar.update(1) 




