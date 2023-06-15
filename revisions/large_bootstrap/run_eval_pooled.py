import json
import fire
import os
import sys
from tqdm import tqdm
sys.path.append('../..')
from patient_evaluation import main as patient_eval
from joblib import Parallel, delayed

def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def main(
    pred_folder = 'pooled_prediction_bootstraps', # input
    eval_folder = 'evaluation_pooled_prediction_bootstraps', # output
    num_steps = 100, #default
    lambda_path='../../config/lambdas', # old args from here
    n_jobs=1,
    cost=5,
    drop_percentiles=True,
    from_dict=True,
    ):
    
    os.makedirs(eval_folder, exist_ok=True)
    
    datasets = ['aumc', 'hirid', 'mimic', 'eicu']
    n_reps = 5; n_subsamples = 10
    n_bootstraps = 40
    kwargs = {'num_steps': num_steps, 'drop_percentiles': drop_percentiles, 'from_dict': from_dict,
		'n_jobs': n_jobs, 'cost': cost, 'lambda_path': lambda_path}

    # generate all arguments
    arguments = [(dataset_train, dataset_eval, rep, subsample, pred_folder, n_bootstraps, eval_folder, kwargs) 
                for dataset_train in ['pooled'] 
                for dataset_eval in datasets 
                for rep in range(n_reps) 
                for subsample in range(n_subsamples)]

    # run in parallel
    eval_n_jobs = 100  # set this to the number of cores you want to use, -1 means using all processors
    _ = Parallel(n_jobs=eval_n_jobs)(delayed(process_sample)(*args) for args in arguments)


    #total = len(datasets)**2 * n_reps * n_subsamples * n_bootstraps
    #pbar = tqdm(total=total, desc="Progress")
    #for dataset_train in datasets:
    #    for dataset_eval in datasets:
    #        for rep in range(n_reps):
    #            for subsample in range(n_subsamples):

    #                bt_file = os.path.join(
    #                        pred_folder,
    #                        f'bootstrap_pred_{dataset_train}_{dataset_eval}_rep_{rep}_subsample_{subsample}.json'
    #                )
    #                bt_preds = load_json(bt_file)
    #                
    #                bt_results = []

    #                for i in range(n_bootstraps):
    #                    
    #                    input_file = bt_preds[i] 
    #                    
    #                    # get input and output file
    #                    # run eval:
    #                    result = patient_eval(
    #                        input_file, # here a dict!
    #                        None, #output file empty 
    #                        num_steps,
    #                        lambda_path,
    #                        n_jobs,
    #                        cost,
    #                        from_dict,
    #                        drop_percentiles,
    #                    )
    #                    bt_results.append(result)
    #                output_file = f'bootstrap_eval_{dataset_train}_{dataset_eval}_rep_{rep}_subsample_{subsample}.json'
    #                output_file = os.path.join(
    #                        eval_folder,
    #                        output_file,
    #                )       
    #                with open(output_file, 'w') as f:
    #                    json.dump(bt_results, f)


def process_sample(dataset_train, dataset_eval, rep, subsample, pred_folder, n_bootstraps, eval_folder, kwargs):
  	  
    bt_file = os.path.join(
            pred_folder,
            f'bootstrap_pred_{dataset_train}_{dataset_eval}_rep_{rep}_subsample_{subsample}.json'
    )
    bt_preds = load_json(bt_file)
	
    bt_results = []

    for i in tqdm(range(n_bootstraps)):
		
        input_file = bt_preds[i] 
		
        # get input and output file
        # run eval:
        result = patient_eval(
            input_file, # here a dict!
            None, #output file empty 
            kwargs['num_steps'],
            kwargs['lambda_path'],
            kwargs['n_jobs'],
            kwargs['cost'],
            kwargs['from_dict'],
            kwargs['drop_percentiles'],
        )
        bt_results.append(result)

    output_file = f'bootstrap_eval_{dataset_train}_{dataset_eval}_rep_{rep}_subsample_{subsample}.json'
    output_file = os.path.join(
            eval_folder,
            output_file,
    )       
    with open(output_file, 'w') as f:
        json.dump(bt_results, f)

    return None 

if __name__ == "__main__":

    fire.Fire(main)
