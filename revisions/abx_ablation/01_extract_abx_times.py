import pandas as pd
import os
from IPython import embed
from tqdm import tqdm 
import fire
import json

def get_abx_time(pat):
    """
    Grep first true in abx column and return stay_time
    """
    assert count_abx(pat) > 0

    inds = ~pat['abx'].isnull() # non-Nan entries
    subset = pat[inds]
    assert subset['abx'].iloc[0] == True
    time = subset['stay_time'].iloc[0]
    return time

def count_abx(pat):
    return (pat['abx'] == True).sum()

def main(dataset='aumc'):

    path=f'/Volumes/Mephisto/PhD/multicenter-sepsis/multicenter-sepsis/datasets/downloads/{dataset}-0-4-0.parquet'
    #path = f'/Volumes/Mephisto/PhD/multicenter-sepsis/multicenter-sepsis/datasets/{dataset}/data/parquet/features_small_cache/{split}_0_cost_5.parquet' 
    #path = '/Volumes/Mephisto/PhD/multicenter-sepsis/multicenter-sepsis/datasets/aumc/data/parquet/aumc.parquet'

    columns = ['stay_id', 'stay_time', 'abx']
    df = pd.read_parquet(path, columns=columns)
    embed();sys.exit()

    abx_times = {}
    for i, pat in tqdm(df.groupby('stay_id')):
        if count_abx(pat) > 0:
            time = get_abx_time(pat)
            stay_id = pat['stay_id'].iloc[0]
            assert stay_id not in abx_times.keys()
            abx_times[int(stay_id)] = time
    with open(f'abx_times_{dataset}.json', 'w') as f:
        json.dump(abx_times, f)

            



if __name__ == "__main__":
    datasets = ['aumc', 'hirid','eicu']
    for dataset in datasets:
        print(f'Processing {dataset}..')
        main(dataset)
    #fire.Fire(main)


