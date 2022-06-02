"""Script to create smaller splits of the finetuning (validation) split"""
import json
import os
from IPython import embed
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def read_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def main():

    datasets = ['aumc', 'mimic', 'hirid', 'eicu']
    test_sizes = [0.5, 0.8] #corresponds to 0.5 and 0.2 train_size
    split_path = 'config/splits' 

    for dataset in datasets:
        d = read_json(
            os.path.join(
                split_path,
                f'splits_{dataset}.json'
            ) 
        )

        ids  = d['total']['ids']
        labels = d['total']['labels']
        id_to_labels = {x: y for x,y in zip(ids, labels)}
       
        # we subdivide starting from the split_0 val split:
        ids_start = d['dev']['split_0']['validation'] 
        labels_start = [id_to_labels[x] for x in ids_start]
        for test_size in test_sizes:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
            sss.get_n_splits(labels_start)
            for train_index, test_index in sss.split(labels_start,labels_start):
                train_ids = np.array(ids_start)[train_index]
                d['dev']['split_0'][f'finetuning_train_size_{round(1-test_size,3)}'] = train_ids.tolist()
                l = [id_to_labels[x] for x in train_ids.tolist()]
                print(np.sum(l) / len(l))
                # had to round due to weird display of 0.2 as 0.1999..
            
        output_path = os.path.join(
            split_path,
            'finetuning',
            f'splits_{dataset}.json' 
        )
        print(f'Writing finetuning split to {output_path}') 
        with open(output_path, 'w') as f:
            json.dump(d, f, indent=4)


if __name__ == "__main__":

    main()
