import pytest
import pyarrow.parquet as pq
import numpy as np

@pytest.mark.parametrize(
    'dataset_path',
    [
        'datasets/physionet2019/data/parquet/features',
        'datasets/mimic_demo/data/parquet/features',
        'datasets/mimic/data/parquet/features',
        'datasets/hirid/data/parquet/features',
        'datasets/eicu/data/parquet/features',
        'datasets/aumc/data/parquet/features'
    ]
)
def test_consecutive_sorted_ids(dataset_path):
    table = pq.read_table(dataset_path, columns=['stay_id', 'stay_time'])
    assert np.all(np.diff(table['stay_id'].to_numpy()) > 0)
