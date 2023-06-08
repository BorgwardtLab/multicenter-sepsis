import pandas as pd
import os
from IPython import embed
from tqdm import tqdm 
import fire
import json

def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)


def main(dataset='aumc',sepsis_case=True):
    
    # we want to filter to for sepsis cases or controls.

    #pred_path = '/Volumes/Mephisto/PhD/multicenter-sepsis/multicenter-sepsis/revisions/results/evaluation_test/internal_preds_for_drago_temporal_analysis/prediction_output_subsampled' 
    disk_root = '/Volumes/Mephisto/PhD/multicenter-sepsis/multicenter-sepsis/'
    thres_path = os.path.join(disk_root, 'revisions/results/evaluation_test/internal_preds_for_drago_temporal_analysis/evaluation_results_at_80recall.csv')
    #pred_file = os.path.join(pred_path, 'aumc_rep_0_j76ft4wm_AUMC_subsample_0.json')
    pred_mapping_file = os.path.join(disk_root, 'revisions/results/evaluation_test/prediction_subsampled_mapping.json') 
    abx_times_file = f'abx_times/abx_times_{dataset}.json'

    # load data:
    # preds = load_json(pred_file)
    abx_times = load_json(abx_times_file)
    df = pd.read_csv(thres_path) # threshold table
    pred_mapping = load_json(pred_mapping_file)
    pred_mapping = pred_mapping['AttentionModel']

    # Internal results, so train_dataset == test_dataset
    for (dataset_train, rep, subsample), df_ in tqdm(df.groupby(['dataset_train', 'rep', 'subsample'])):
        assert len(df_) == 1
        rep = int(rep)
        subsample = int(subsample) # comes as float out of df

        #TODO: customize this for other than internal results: 
        pred_file = pred_mapping[dataset_train][dataset_train][f'rep_{rep}'][f'subsample_{subsample}']
        preds = load_json(
                os.path.join(disk_root, pred_file)
        )
        # get current threshold:
        try:
            curr_thres = df_['thres'].iloc[0]
        except:
            embed()
        ###  process single run (dataset, rep, subsample)

        patients = {}
        # Fetch all cases (or controls) depending on sepsis_case
        for pat_index, stay_id in tqdm(enumerate(preds['ids'])):
            is_case = sum(preds['labels'][pat_index]) > 0

            if not is_case == sepsis_case:
                # if is_case doesnt match our filter we skip
                continue
            
            # processing patient
            # for each time step:
            d = {} # patient dict
            if str(stay_id) in abx_times.keys(): 
                abx_time = abx_times[str(stay_id)]
            else:
                # no abx received, set its time very large number
                abx_time = 10**8

            alarm_received = 0
            for time_index, time in enumerate(preds['times'][pat_index]):
                dt = {} # time step dict
                dt['labels'] = preds['labels'][pat_index][time_index]
                # 1) did patient already receive AB?
                dt['abx'] = time >= abx_time
                dt['scores'] = preds['scores'][pat_index][time_index]
                # 2) did patient receive alarm now or previously?
                # check if score >= current threshold
                if dt['scores'] >= curr_thres:
                    alarm_received = 1
                dt['alarm_received'] = alarm_received
                dt['thres'] = curr_thres
                d[time] = dt

            assert stay_id not in patients.keys()
            patients[stay_id] = d
            
            #if pat_index == 10:
            #    embed(); sys.exit()
        embed();sys.exit()        

        # Cache current dict to json
        out_file = f'output/abx_analysis_cache_{dataset_train}_rep_{rep}_subsample_{subsample}.json'
        #assert not os.path.exists(out_file)
        with open(out_file, 'w') as f:
            json.dump(patients, f, indent=4)

        

if __name__ == "__main__":
    fire.Fire(main)


