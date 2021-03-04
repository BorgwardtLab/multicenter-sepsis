"""Dataset loading."""
import bisect
import os
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch.utils.data import Dataset
import json


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

        self._build_id_lookup()

    def _build_id_lookup(self):
        ids_sorted = sorted(ids)

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
            return table.to_pandas()
        return table


if __name__ == '__main__':
    import argparse
    from random import Random
    import timeit
    parser = argparse.ArgumentParser()
    parser.add_argument('parquet_dataset', type=str)
    parser.add_argument('--split_file', type=str, required=True)
    args = parser.parse_args()

    with open(args.split_file, 'r') as f:
        ids = json.load(f)['total']['ids']

    n_ids = len(ids)

    dataset = ParquetDataset(
        args.parquet_dataset, ids, id_column='stay_id')
    rand = Random()
    time = timeit(
        'index=rand.randint(0, n_ids-1); dataset[index]',
        number=100,
        globals={
            'rand': rand,
            'n_ids': n_ids,
            'dataset': dataset
        }
    )
    print(time)
