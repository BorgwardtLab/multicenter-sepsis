import pandas as pd
import os
from IPython import embed
from tqdm import tqdm 
import fire
import json

def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def main(dataset='aumc'):
    
    # we want to filter to for sepsis cases or controls.

    #pred_path = '/Volumes/Mephisto/PhD/multicenter-sepsis/multicenter-sepsis/revisions/results/evaluation_test/internal_preds_for_drago_temporal_analysis/prediction_output_subsampled' 
    disk_root = '/Volumes/Mephisto/PhD/multicenter-sepsis/multicenter-sepsis/'
    pred_path = disk_root + 'revisions/results/evaluation_test/internal_preds_for_drago_temporal_analysis/prediction_output_subsampled'
    pred_file = os.path.join(pred_path, 'aumc_rep_0_j76ft4wm_AUMC_subsample_0.json')
    #thres_path = os.path.join(disk_root, 'revisions/results/evaluation_test/internal_preds_for_drago_temporal_analysis/evaluation_results_at_80recall.csv')
    
    pred_mapping_file = os.path.join(disk_root, 'revisions/results/evaluation_test/prediction_subsampled_mapping.json') 

    # load data:
    #preds = load_json(pred_file)
    pred_mapping = load_json(pred_mapping_file)
    pred_mapping = pred_mapping['AttentionModel']
    
    datasets = ['aumc', 'hirid', 'mimic', 'eicu']
    for dataset_train in datasets:
        for dataset_eval in datasets:
            for rep in range(5):
                for subsample in range(10):

                    #TODO: customize this for other than internal results: 
                    pred_file = pred_mapping[dataset_train][dataset_eval][f'rep_{rep}'][f'subsample_{subsample}']
                    preds = load_json(
                            os.path.join(disk_root, pred_file)
                    )

                    embed() ; sys.exit()

        

if __name__ == "__main__":
    fire.Fire(main)


