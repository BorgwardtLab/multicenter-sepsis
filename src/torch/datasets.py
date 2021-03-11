"""Dataset loading."""
import bisect
import os
import pathlib
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
from torch.utils.data import Dataset
import json

from src.variables.feature_groups import ColumnFilterLight
from src.variables.mapping import VariableMapping

VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


class Normalize:
    """Transform for normalizing some of the datasets input columns."""

    def __init__(self, normalization_config):
        with open(normalization_config, 'r') as f:
            d = json.load(f)

        self.mean = pd.Series(d['means'])
        self.std = pd.Series(d['stds'])


    def __call__(self, df):
        df = df.subtract(self.mean, axis=1, fill_value=0.)
        return df.div(self.std, axis=1, fill_value=1.)


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
            ** self.reader_args
        )
        if self.as_pandas:
            return table.to_pandas(self_destruct=True, ignore_metadata=True)
        return table


class SplittedDataset(ParquetDataset):
    """Dataset with predefined splits and feature groups."""

    ID_COLUMN = VM_DEFAULT('id')
    TIME_COLUMN = VM_DEFAULT('time')
    # TODO: It looks like age and sex are not present in the data anymore
    STATIC_COLUMNS = VM_DEFAULT.all_cat('static') #['weight', 'height']

    def __init__(self, path, split_file, split, feature_set,
                 only_physionet_features=False, fold=0, pd_transform=None, transform=None):
        with open(split_file, 'r') as f:
            if split in ['train', 'validation']:
                ids = json.load(f)['dev']['split_{}'.format(fold)][split]
            else:
                ids = json.load(f)[split]['split_{}'.format(fold)]

        super().__init__(
            path, ids, self.ID_COLUMN, columns=None, as_pandas=True)

        if only_physionet_features:
            self.columns = ColumnFilterLight(
                self._dataset_columns).physionet_set(feature_set=feature_set)
        else:
            self.columns = ColumnFilterLight(
                self._dataset_columns).feature_set(name=feature_set)
        self.pd_transform = transform
        self.transform = transform

    def __getitem__(self, index):
        df = super().__getitem__(index)
        if self.pd_transform:
            df = self.pd_transform(df)
        id = df[self.ID_COLUMN].values[0]
        times = df[self.TIME_COLUMN].values
        statics = df[self.STATIC_COLUMNS].values[0]
        ts = df.drop(
            columns=[self.ID_COLUMN, self.TIME_COLUMN] + self.STATIC_COLUMNS)
        out = {
            'id': id,
            'times': times,
            'statics': statics,
            'ts': ts.values
        }

        if self.transform:
            out = self.transform(out)
        return out


class Physionet2019(SplittedDataset):
    """Physionet 2019 dataset."""

    def __init__(self, split, feature_set, only_physionet_features=True, fold=0):
        super().__init__(
            'datasets/physionet2019/data/parquet/features',
            'config/splits/splits_physionet2019.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold
        )


class MIMICDemo(SplittedDataset):
    """MIMIC demo dataset."""

    def __init__(self, split, feature_set, only_physionet_features=False, fold=0):
        normalizer = Normalize(
            'config/normalizer/normalizer_mimic_demo_rep_{}.json'.format(fold))
        super().__init__(
            'datasets/mimic_demo/data/parquet/features',
            'config/splits/splits_mimic_demo.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            pd_transform=normalizer
        )


class MIMIC(SplittedDataset):
    """MIMIC dataset."""

    def __init__(self, split, feature_set, only_physionet_features=False, fold=0):
        normalizer = Normalize(
            'config/normalizer/normalizer_mimic_rep_{}.json'.format(fold))
        super().__init__(
            'datasets/mimic/data/parquet/features',
            'config/splits/splits_mimic.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            pd_transform=normalizer
        )


class Hirid(SplittedDataset):
    """Hirid dataset."""

    def __init__(self, split, feature_set, only_physionet_features=False, fold=0):
        normalizer = Normalize(
            'config/normalizer/normalizer_hirid_rep_{}.json'.format(fold))
        super().__init__(
            'datasets/hirid/data/parquet/features',
            'config/splits/splits_hirid.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            pd_transform=normalizer
        )


class EICU(SplittedDataset):
    """EICU dataset."""

    def __init__(self, split, feature_set, only_physionet_features=False, fold=0):
        normalizer = Normalize(
            'config/normalizer/normalizer_eicu_rep_{}.json'.format(fold))
        super().__init__(
            'datasets/eicu/data/parquet/features',
            'config/splits/splits_eicu.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            pd_transform=normalizer
        )


class AUMC(SplittedDataset):
    """AUMC dataset."""

    def __init__(self, split, feature_set, only_physionet_features=False, fold=0):
        normalizer = Normalize(
            'config/normalizer/normalizer_aumc_rep_{}.json'.format(fold))
        super().__init__(
            'datasets/aumc/data/parquet/features',
            'config/splits/splits_aumc.json',
            split,
            feature_set,
            only_physionet_features=only_physionet_features,
            fold=fold,
            pd_transform=normalizer
        )


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
