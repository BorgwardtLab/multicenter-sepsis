"""This script allows to pool predictions on a dataset using models trained on all other datasets """
import argparse
import json
from IPython import embed
from collections import defaultdict
import os 
import pandas as pd

class PredictionLoader():
    """Helper class for handling prediction files """
    def __init__(self, path):
        self.d = self._load(path)
    
    def _load(self, fname):
        with open(fname, 'r') as f:
            return json.load(f)
 
    def __call__(self, model, d_tr, d_ev, rep, ss):
        """
        get pred file for current model, dataset_train,
        dataset_eval, repetition and subsample
        """  
        fname = self.d[model][d_tr][d_ev]['rep_' + str(rep)]['subsample_'+str(ss)]
        return self._load(fname)

    def extract_preds(self, model, train_datasets, dataset_eval, rep, subsample):
        """ extract predictions of all train dataset for 
            current eval dataset, rep, and subsample.
            also return one input dict for creating output dict later on.    
        """
        dfs = []
        for train_dataset in train_datasets:
            pred_dict = self.__call__(model, train_dataset, dataset_eval, rep, subsample)
           
            pred_df = self._to_df(pred_dict) 
            dfs.append(pred_df)

        df = self._merge_dfs(dfs)

        return df, pred_dict

    def _to_df(self, d):
        """convert dict containing scores to df"""
        data = defaultdict(list)
        for (i, times, labels, scores) in zip(d['ids'],
            d['times'], d['labels'], d['scores']):
            for time, label, score in zip(times, labels, scores):
                data['stay_time'].append(time)
                data['labels'].append(label)
                data['scores'].append(score)
                data['stay_id'].append(i) 
        df = pd.DataFrame(data) 
        # add static information
        keys = ['rep', 'subsample', 'dataset_eval', 'dataset_train']
        for key in keys:
            df[key] = d[key]
        return df  

    def _merge_dfs(self, dfs):
        """ merge list of dfs into one and check index consistency"""
        out = []
        index_keys = ['stay_id','stay_time', 'labels','rep','subsample','dataset_eval'] 
        for df in dfs:
            df = df.set_index(index_keys)
            if len(out) > 0:
                assert (df.index != out[0].index).sum() == 0 #check that multi index is identical 
            df = df.rename(columns={'scores': df['dataset_train'].iloc[0]})
            df = df.drop(columns=['dataset_train'])
            out.append(df)
         
        out_df = pd.concat(out, axis=1)
 
        return out_df

def to_dict(df, d_in):
    """convert pooled df to dictionary """
    keys = ['label_propagation', 'task', 'rep', 'dataset_kwargs', 
        'dataset_eval', 'split', 'ids', 'labels', 'targets',
        'times', 'subsample', 'subsampled_prevalence', 'model'
    ]
    d_out = {key: d_in[key] for key in keys if key in d_in.keys()}
    d_out['dataset_train'] = 'pooled'

    # sanity check:
    df_r = df.reset_index()
    ids = df_r['stay_id'].unique().tolist()
    assert all([x == y for x,y in zip(ids, d_in['ids'])])
    unrolled_times = [x for times in d_in['times'] for x in times ]
    times = df_r['stay_time'].tolist()
    assert all([x == y for x,y in zip(times, unrolled_times)])
    
    # convert scores to list of list
    scores = []
    for i in d_in['ids']:
        scores.append(
            df.loc[i].values.tolist()
        ) 
    d_out['scores'] = scores
    
    return d_out 
    
 
def main(args):
    
    # hard-coded params:
    n_subsamples = 10 #prevalence subsampling
    n_reps = 5 # repetition folds
    
    output_path = args.output_path

    model = args.model
    dataset_eval = args.dataset_eval
    datasets = ['mimic','eicu','hirid','aumc']
    if args.add_emory:
        datasets.append('physionet2019')
    train_datasets = datasets.copy()
    train_datasets.remove(dataset_eval)

    pl = PredictionLoader(args.mapping_file)

    preds_list = [] 
    for rep in range(n_reps):
        for subsample in range(n_subsamples):
            preds, input_dict = pl.extract_preds(model, train_datasets, dataset_eval, rep, subsample)
            preds_list.append(preds)

            for fn in ['max']: #'mean'
                pooled = getattr(preds, fn)(axis=1) 
                d = to_dict(pooled, input_dict)
                
                # write pooled dict out
                pooled_path = os.path.join(output_path, fn, 'prediction_output')
                os.makedirs(pooled_path, exist_ok=True)
                fname = f'preds_{fn}_pooled_{model}_{dataset_eval}_rep_{rep}_subsample_{subsample}'
                if args.add_emory:
                    fname += '_w_emory'
                fname += '.json'
                with open(os.path.join(pooled_path, fname), 'w') as f:
                    json.dump(d, f)
                 
    df = pd.concat(preds_list)
    raw_file = f'raw_scores_{model}_{dataset_eval}'
    if args.add_emory:
        raw_file += '_w_emory'
    raw_file += '.csv'
    df.to_csv(os.path.join(output_path, raw_file))
 

if __name__ == '__main__': 
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file', help='json which maps models and datasets to prediction files',
        default='results/evaluation_test/prediction_mapping.json')
    parser.add_argument('--dataset_eval', help='predictions on which dataset')
    parser.add_argument('--model', help='which model to pool', default='AttentionModel')
    parser.add_argument('--add_emory', help='add Emory to pooling exp (different setup)', action='store_true')
    parser.add_argument('--output_path', help='path where pooled preds are written to')
 
    args = parser.parse_args() 
    main(args)
