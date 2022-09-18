"""Script to create larger splits of the finetuning (validation) split"""
import json
import os
from IPython import embed
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def read_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def main():
    #TODO: 
    #  - load finetuning split
    #  -  add n-10% from train split
    #  -  create sets with 20,30,40, 50 % 
    
    datasets = ['aumc', 'mimic', 'hirid', 'eicu']
    #test_sizes = [0.5, 0.8] #corresponds to 0.5 and 0.2 train_size
    finetuning_sizes = np.array([0.2,0.3,0.4,0.5]) # for comparability with ft < 10% we 
    # reuse validation split + train split for remainder
    portion_from_train_split = finetuning_sizes - 0.1 # percentage of overall dataset used from trainn split
    ratio_of_train_split = portion_from_train_split / 0.8 # due to train spli making up 80%
    test_sizes = 1- ratio_of_train_split # test_size to apply on train split 
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
       
        # we subdivide starting from the split_0 train split:
        ids_start = d['dev']['split_0']['train'] 
        labels_start = [id_to_labels[x] for x in ids_start]
        # and addd val split to each newly created split (to be comparable with ft splits created from val split)
        ids_val = d['dev']['split_0']['validation'] 
        labels_val = [id_to_labels[x] for x in ids_val]

        for test_size, ft_size in zip(test_sizes, finetuning_sizes):
            print(f'Creating split for test size {test_size}')
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
            sss.get_n_splits(labels_start)
            for train_index, test_index in sss.split(labels_start,labels_start):
                train_ids = np.array(ids_start)[train_index]
                train_ids = ids_val + train_ids.tolist()
                d['dev']['split_0'][f'finetuning_train_size_{ft_size*10}'] = train_ids
                l = [id_to_labels[x] for x in train_ids]
                print(np.sum(l) / len(l))
                # had to round due to weird display of 0.2 as 0.1999..
        output_path = os.path.join(
            split_path,
            'finetuning_large',
            f'splits_{dataset}.json' 
        )
        print(f'Writing finetuning split to {output_path}') 
        with open(output_path, 'w') as f:
            json.dump(d, f, indent=4)


if __name__ == "__main__":

    main()
