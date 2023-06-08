import json
import glob
from IPython import embed
import numpy as np
import pandas as pd

def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def process_run(d, abx_times, max_hours=100):
    results = {}
    for pid, pat in d.items():
        alarm_time = None
        for time in pat.keys():
            data = pat[time]
            if data['alarm_received']:
                alarm_time = int(float(time)) # weird format
                #print(f'alarm found for {pid} at time {time}')
                break
        results[pid] = {
            'abx_time': abx_times[pid] if pid in abx_times.keys() else None,
            'alarm_time': alarm_time,
        }

    total = 0
    n_alarm_before_abx = 0
    n_alarm_no_abx = 0
    n_alarm_not_before_abx = 0
    problem_ids = []
    abx_earliness = []
    for i, data in results.items():
        t_alarm, t_abx = data['alarm_time'], data['abx_time']
        total += 1
        if t_alarm is None:
            continue # not interesting case
        if t_abx is None:
            #print(f'Problem: ID {i} has no abx time!')
            problem_ids.append(i)
            n_alarm_no_abx += 1
        else: 
            abx_earliness.append(
                t_alarm - t_abx
            )
            if t_abx > t_alarm:
                n_alarm_before_abx += 1
            else:
                n_alarm_not_before_abx +=1

    counts = {'total': total, 
            'n_alarm_before_abx': n_alarm_before_abx, 
            'n_alarm_no_abx': n_alarm_no_abx,
            'n_alarm_not_before_abx': n_alarm_not_before_abx
    }
    print(counts)
    mean_earli = np.mean(abx_earliness)
    print('mean earliness:', mean_earli)
    counts['mean_abx_earli'] = mean_earli
    #path=f'/Volumes/Mephisto/PhD/multicenter-sepsis/multicenter-sepsis/datasets/downloads/{dataset}-0-4-0.parquet'
    #df = pd.read_parquet(path) #, columns=columns

    #embed()
    return counts 

def process_run_time_first(d, max_hours=100):
    results = {}
    for time in np.arange(0, max_hours):

        total = 0
        n_alarms = 0
        n_abx = 0
        n_onset = 0

        for pid, pat in d.items():
            # check if pat has this time:
            time = str(float(time)) # due to weird format '0.0' in dict 
            if time in pat.keys():
                data = pat[time]
                
                n_onset += data['labels'] # concurrent or post onset
                n_abx += int(data['abx'])
                n_alarms += data['alarm_received']
                total += 1
        
        results[time] = {
            'total': total,
            'n_alarms': n_alarms,
            'n_abx': n_abx,
            'n_onset': n_onset,
        }

    embed();sys.exit()


files = glob.glob('output/*.json')

datasets=['aumc','mimic','hirid','eicu']
repetitions = np.arange(5)
subsamples = np.arange(10)

stats = {}
flat_stats = []
for dataset in datasets:
    if dataset not in stats.keys():
        stats[dataset] = {}

    abx_times_file = f'abx_times/abx_times_{dataset}.json'
    abx_times = load_json(abx_times_file)
      
    for rep in repetitions:
        if rep not in stats[dataset].keys():
            stats[dataset][rep] = {}
        for subsample in subsamples:
            input_file = f'output/abx_analysis_cache_{dataset}_rep_{rep}_subsample_{subsample}.json'
            with open(input_file, 'r') as f:
                d = json.load(f)
            # process file
            current_stats = process_run(d, abx_times)
            stats[dataset][rep][subsample] = current_stats
            flat_stats.append(
                    {'dataset': dataset, 'rep': rep, 'subsample': subsample, **current_stats}
            )
# flatten dict for df:
df = pd.DataFrame(flat_stats)

def get_rel_scores(df):
    before = df['n_alarm_before_abx'] + df['n_alarm_no_abx']
    after = df['n_alarm_not_before_abx']
    before_rel = before / (before + after)
    print(f'Fraction of TP alarms before {before_rel.mean()}')
    return before_rel


for dataset, df_ in df.groupby('dataset'):
    print(dataset)
    _  = get_rel_scores(df_)

print('Overall')
before_rel = get_rel_scores(df)

embed(); sys.exit()


