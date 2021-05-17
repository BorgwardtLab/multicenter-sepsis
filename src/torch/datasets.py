"""Dataset loading."""
import bisect
import json
import os
from itertools import accumulate
import operator
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.torch.torch_utils import ComposeTransformations

from src.variables.feature_groups import ColumnFilterLight
from src.variables.mapping import VariableMapping

VM_CONFIG_PATH = str(
    Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)

__all__ = [
    'Physionet2019',
    'MIMICDemo',
    'MIMIC',
    'Hirid',
    'EICU',
    'AUMC'
]


class Normalize:
    """Transform for normalizing some of the datasets input columns."""

    def __init__(self, normalization_config, columns):
        with open(normalization_config, 'r') as f:
            d = json.load(f)

        means = pd.Series(d['means'])
        # Get the subset of columns that are relevant for us
        norm_cols = list(set(means.index).intersection(columns))
        self.mean = means.loc[norm_cols]
        self.std = pd.Series(d['stds']).loc[norm_cols]

    def __call__(self, df):
        norm_df = df[self.mean.index]
        norm_df = (norm_df - self.mean) / self.std
        df[norm_df.columns] = norm_df
        return df

class ApplyLambda:
    """ Apply lambda to regression target """
    def __init__(self, lambda_path):
        with open(lambda_path, 'r') as f:
            self.lam = json.load(f)['lam']

    def __call__(self, df):
        u = df[VM_DEFAULT('utility')]
        # patient-level label:
        timestep_label = df[VM_DEFAULT('label')]
        l = timestep_label.sum() > 0 #more nan-stable than .any()
        l = l.astype(int)
        #l = l.reindex(u.index) # again timestep wise label
        # applying lambda to target: if case: times lam, else no change
        # need to drop target instead of overwriting as otherwise pandas 
        # throwed an error due to reindexing with duplicate indices..
        df = df.drop(columns=[VM_DEFAULT('utility')])
        new_target = l*u*self.lam + (1-l)*u 
        df[VM_DEFAULT('utility')] = new_target
        return df

class LoadLambda:
    """ Load lambda to regression target """
    def __init__(self, lambda_path):
        with open(lambda_path, 'r') as f:
            self.lam = json.load(f)['lam']


class Impute:
    """ Transform to impute and remove invalid values"""
    def __init__(self, fill_value=0):
        self.fill_value = fill_value
    def __call__(self, df):
        df = df.fillna(self.fill_value)
        # sanity check if there is degenerate normalization
        df = df.replace([np.inf, -np.inf], 0)
        return df
        

class ParquetDataset(Dataset):
    METADATA_FILENAME = '_metadata'

    def __init__(self, path, ids, id_column, columns=None, as_pandas=False, **kwargs):
        super().__init__()
        self.path = path
        self.ids = ids
        self.id_column = id_column
        self.columns = columns
        self.as_pandas = as_pandas
        self.reader_args = kwargs

        self._parse_and_store_metadata()

    def _parse_and_store_metadata(self):
        ids_sorted = sorted(self.ids)

        metadata = pq.read_metadata(
            os.path.join(self.path, self.METADATA_FILENAME))
        id_column_index = metadata.schema.names.index(self.id_column)
        id_to_file_mapping = {}
        # Iterate over the row groups, we might get each id more than once, as
        # it can be split across multiple row groups. That should not matter
        # though as we assume a single id is not distributed over more than one
        # file.
        for i in range(metadata.num_row_groups):
            column = metadata.row_group(i).column(id_column_index)
            cur_file = column.file_path
            stats = column.statistics
            rg_min, rg_max = stats.min, stats.max
            # Find all ids which are spanned by this row group add them to the
            # lookup dict
            begin = bisect.bisect_left(ids_sorted, rg_min)
            end = bisect.bisect_right(ids_sorted, rg_max)
            for i in range(begin, end):
                cur_id = ids_sorted[i]
                if cur_id in id_to_file_mapping.keys():
                    # Ensure our assumption of ids not being distributed
                    # across different files holds.
                    assert id_to_file_mapping[cur_id] == cur_file
                else:
                    id_to_file_mapping[cur_id] = cur_file

        self._dataset_columns = metadata.schema.names
        self._id_to_file_lookup = id_to_file_mapping

    def __getitem__(self, index):
        item_id = self.ids[index]
        file = self._id_to_file_lookup[item_id]

        table = pq.read_table(
            os.path.join(self.path, file),
            columns=self.columns,
            use_legacy_dataset=False,  # Needed for filtering
            filters=[(self.id_column, '=', item_id)],
            use_pandas_metadata=True if self.as_pandas else False,
            **self.reader_args
        )
        if self.as_pandas:
            return table.to_pandas(self_destruct=True, ignore_metadata=True)
        return table
    
    def __len__(self):
        return len(self.ids)

class ParquetLoadedDataset(Dataset):
    """
    In this version, we load the entire dataset into memory for speed.
    """
    def __init__(self, path, ids, id_column, columns=None, **kwargs):
        super().__init__()
        self.path = path
        self.ids = ids
        self.id_column = id_column
        self.columns = columns
        self.data = self._load_dataset()
        self._dataset_columns = self.data.columns

    def _load_dataset(self):
        df = pd.read_parquet(self.path)
        return df 

    def __getitem__(self, index):
        item_id = self.ids[index]
        return self.data.loc[item_id]
    
    def __len__(self):
        return len(self.ids)



class SplittedDataset(ParquetLoadedDataset):
    """Dataset with predefined splits and feature groups."""

    ID_COLUMN = VM_DEFAULT('id')
    TIME_COLUMN = VM_DEFAULT('time')
    STATIC_COLUMNS = VM_DEFAULT.all_cat('static')
    LABEL_COLUMN = VM_DEFAULT('label')
    UTILITY_COLUMN = VM_DEFAULT('utility')

    def __init__(self, path, split_file, split, feature_set,
                 only_physionet_features=False, fold=0, pd_transform=None, transform=None):
        with open(split_file, 'r') as f:
            d = json.load(f)
            if split in ['train', 'validation']:
                ids = d['dev']['split_{}'.format(fold)][split]
            else:
                ids = d[split]['split_{}'.format(fold)]
            # Need this to construct stratified split
            self.id_to_label = dict(zip(d['total']['ids'], d['total']['labels']))

        super().__init__(
            path, ids, self.ID_COLUMN, columns=None, as_pandas=True)

        if only_physionet_features:
            self.columns = ColumnFilterLight(
                self._dataset_columns).physionet_set(feature_set=feature_set)
            self.STATIC_COLUMNS = [col for col in self.STATIC_COLUMNS if col in self.columns]
        else:
            self.columns = ColumnFilterLight(
                self._dataset_columns).feature_set(name=feature_set)
        self.columns = self.columns + [self.LABEL_COLUMN, self.UTILITY_COLUMN]
        self.pd_transform = pd_transform
        self.transform = transform

    def get_stratified_split(self, random_state=None):
        per_instance_labels = [self.id_to_label[id] for id in self.ids]
        train_indices, test_indices = train_test_split(
            range(len(per_instance_labels)),
            train_size=0.9,
            stratify=per_instance_labels,
            random_state=random_state
        )
        return train_indices, test_indices

    def __getitem__(self, index):
        df = super().__getitem__(index)[self.columns]
        if self.pd_transform:
            df = self.pd_transform(df)
        id = df.index.values[0]
        times = df[self.TIME_COLUMN].values
        statics = df[self.STATIC_COLUMNS].values[0]
        drop_cols = [self.TIME_COLUMN, self.LABEL_COLUMN, self.UTILITY_COLUMN] + self.STATIC_COLUMNS
        # ParquetLoadedDataset has id in the index
        # therefore we need to drop id only for base
        # classes where this is not the case:
        if self.__class__.__base__.__base__.__name__ == 'ParquetDataset':
            drop_cols.append(self.ID_COLUMN) 
            
        ts = df.drop(
            columns=drop_cols)
        out = {
            'id': id,
            'times': times,
            'statics': statics,
            'ts': ts.values,
            'labels': df[self.LABEL_COLUMN].values,
            'targets': df[self.UTILITY_COLUMN].values
        }

        if self.transform:
            out = self.transform(out)
        return out

    @property
    def class_imbalance_factor(self):
        if hasattr(self, 'data'):
            prev = self.data[VM_DEFAULT('label')].sum()/len(self.data)
            cif = (1 - prev) / prev
            print(f'Using imbalance factor: {cif}') 
            return cif 
        else:
            print("""class imbalance factor requires dataset to be loaded 
                (ParquetLoadedDataset base class), ignoring class imbalance
                and using 1 as imbalance factor""") 
            return 1

class Physionet2019(SplittedDataset):
    """Physionet 2019 dataset."""

    def __init__(self, split, feature_set='small', only_physionet_features=True, fold=0, cost=5, transform=None):
        super().__init__(
            f'datasets/physionet2019/data/parquet/features_small_cache/{split}_{fold}_cost_{cost}.parquet',
            #'datasets/physionet2019/data/parquet/features_small',
            'config/splits/splits_physionet2019.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            transform=transform
        )
        #normalize = Normalize(
        #    'config/normalizer/normalizer_physionet2019_rep_{}.json'.format(fold),
        #    self.columns
        #)
        
        self.lam = LoadLambda(
            lambda_path =  f'config/lambdas/lambda_physionet2019_rep_{fold}_cost_{cost}.json').lam 
        
        #transforms = [
        #    normalize,
        #    apply_lam,
        #    Impute(),
        #]
        #self.pd_transform = ComposeTransformations(transforms)


class MIMICDemo(SplittedDataset):
    """MIMIC demo dataset."""

    def __init__(self, split, feature_set='small', only_physionet_features=False, fold=0, cost=5, transform=None):
        super().__init__(
            f'datasets/mimic_demo/data/parquet/features_small_cache/{split}_{fold}_cost_{cost}.parquet',
            'config/splits/splits_mimic_demo.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            transform=transform
        )
        #normalize = Normalize(
        #    'config/normalizer/normalizer_mimic_demo_rep_{}.json'.format(fold),
        #    self.columns
        #)
        self.lam = LoadLambda(
            lambda_path =  f'config/lambdas/lambda_mimic_demo_rep_{fold}_cost_{cost}.json' ).lam 
        #transforms = [
        #    normalize,
        #    apply_lam,
        #    Impute(),
        #]
        #self.pd_transform = ComposeTransformations(transforms)

class MIMIC(SplittedDataset):
    """MIMIC dataset."""

    def __init__(self, split, feature_set='small', only_physionet_features=False, fold=0, cost=5, transform=None):
        print(f'Using data fold {fold}...')
        super().__init__(
            f'datasets/mimic/data/parquet/features_small_cache/{split}_{fold}_cost_{cost}.parquet',
            'config/splits/splits_mimic.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            transform=transform
        )
        #normalize = Normalize(
        #    'config/normalizer/normalizer_mimic_rep_{}.json'.format(fold),
        #    self.columns
        #)
        self.lam = LoadLambda(
            lambda_path =  f'config/lambdas/lambda_mimic_rep_{fold}_cost_{cost}.json').lam 
        
        #transforms = [
        #    normalize,
        #    apply_lam,
        #    Impute(),
        #]
        #self.pd_transform = ComposeTransformations(transforms)

class Hirid(SplittedDataset):
    """Hirid dataset."""

    def __init__(self, split, feature_set='small', only_physionet_features=False, fold=0, cost=5, transform=None):
        super().__init__(
            f'datasets/hirid/data/parquet/features_small_cache/{split}_{fold}_cost_{cost}.parquet',
            #'datasets/hirid/data/parquet/features_small',
            'config/splits/splits_hirid.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            transform=transform
        )
        #normalize = Normalize(
        #    'config/normalizer/normalizer_hirid_rep_{}.json'.format(fold),
        #    self.columns
        #)
        self.lam = LoadLambda(
            lambda_path =  f'config/lambdas/lambda_hirid_rep_{fold}_cost_{cost}.json').lam 
        
        #transforms = [
        #    normalize,
        #    apply_lam,
        #    Impute(),
        #]
        #self.pd_transform = ComposeTransformations(transforms)


class EICU(SplittedDataset):
    """EICU dataset."""

    def __init__(self, split, feature_set='small', only_physionet_features=False, fold=0, cost=5, transform=None):
        super().__init__(
            f'datasets/eicu/data/parquet/features_small_cache/{split}_{fold}_cost_{cost}.parquet',
            #'datasets/eicu/data/parquet/features_small',
            'config/splits/splits_eicu.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            transform=transform
        )
        #normalize = Normalize(
        #    'config/normalizer/normalizer_eicu_rep_{}.json'.format(fold),
        #    self.columns
        #)
        self.lam = LoadLambda(
            lambda_path =  f'config/lambdas/lambda_eicu_rep_{fold}_cost_{cost}.json').lam 

        #transforms = [
        #    normalize,
        #    apply_lam,
        #    Impute(),
        #]
        #self.pd_transform = ComposeTransformations(transforms)


class AUMC(SplittedDataset):
    """AUMC dataset."""

    def __init__(self, split, feature_set='small', only_physionet_features=False, fold=0, cost=5, transform=None):
        super().__init__(
            f'datasets/aumc/data/parquet/features_small_cache/{split}_{fold}_cost_{cost}.parquet',
            #'datasets/aumc/data/parquet/features_small',
            'config/splits/splits_aumc.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            transform=transform
        )
        #normalize = Normalize(
        #    'config/normalizer/normalizer_aumc_rep_{}.json'.format(fold),
        #    self.columns
        #)
        self.lam = LoadLambda(
            lambda_path =  f'config/lambdas/lambda_aumc_rep_{fold}_cost_{cost}.json').lam
 
        #transforms = [
        #    normalize,
        #    apply_lam,
        #    Impute(),
        #]
        #self.pd_transform = ComposeTransformations(transforms)


class CombinedDataset(Dataset):
    """Dataset class which combines multiple data sources."""


    def __init__(self, datasets, *args, **kwargs):
        super().__init__()
        self.datasets = [d(*args, **kwargs) for d in datasets]
        self.dataset_lengths = [len(d) for d in self.datasets]
        self.cumsum = [0] + list(accumulate(
            self.dataset_lengths,
            func=operator.add,
        ))

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, index):
        for i, (begin, end) in enumerate(zip(self.cumsum, self.cumsum[1:])):
            if begin <= index < end:
                return self.datasets[i].__getitem__(index - begin)

        raise IndexError(
            f'Index {index} is out of range for CombinedDataset with length {len(self)}')

    @property
    def lam(self):
        """Weighted mean of lambdas of composed datasets."""
        return sum(d.lam * len(d) for d in self.datasets) / self.cumsum[-1]

    @property
    def class_imbalance_factor(self):
        prev = sum(d.data[VM_DEFAULT('label')].sum() for d in self.datasets)
        cif = (1 - prev) / prev
        print(f'Using imbalance factor: {cif}')
        return cif

    def get_stratified_split(self, random_state=None):
        train_indices = []
        test_indices = []
        for offset, dataset in zip(self.cumsum, self.datasets):
            cur_train, cur_test = dataset.get_stratified_split(
                random_state=random_state)
            train_indices.extend(map(lambda i: i + offset, cur_train))
            test_indices.extend(map(lambda i: i + offset, cur_test))
        return train_indices, test_indices


if __name__ == '__main__':
    import argparse
    from random import Random
    from time import time
    from timeit import timeit
    parser = argparse.ArgumentParser()
    parser.add_argument('parquet_dataset', type=str)
    parser.add_argument('--split_file', type=str, required=True)
    args = parser.parse_args()

    with open(args.split_file, 'r') as f:
        ids = json.load(f)['total']['ids']

    n_ids = len(ids)

    start = time()
    dataset = ParquetDataset(
        args.parquet_dataset, ids, id_column='stay_id', as_pandas=True)
    print('Setting up lookup took {} seconds.'.format(time()-start))
    rand = Random()
    n_repetitions = 100
    time = timeit(
        'index=rand.randint(0, n_ids-1); dataset[index]',
        number=n_repetitions,
        globals={
            'rand': rand,
            'n_ids': n_ids,
            'dataset': dataset
        }
    )
    print(
        'Random access to a patient took on average {} seconds'.format(
            time/n_repetitions))
