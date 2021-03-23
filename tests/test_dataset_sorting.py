import pytest
import pyarrow.parquet as pq
import numpy as np


@pytest.mark.parametrize(
    'dataset_path',
    [
        'datasets/downloads/aumc_0.3.0.parquet',
        'datasets/downloads/eicu_0.3.0.parquet',
        'datasets/downloads/eicu_demo_0.3.0.parquet',
        'datasets/downloads/hirid_0.3.0.parquet',
        'datasets/downloads/mimic_0.3.0.parquet',
        'datasets/downloads/mimic_demo_0.3.0.parquet',
        'datasets/downloads/physionet2019_0.3.0.parquet'
    ]
)
def test_sorted_ids_and_times_raw(dataset_path):
    table = pq.read_table(dataset_path, columns=['stay_id', 'stay_time'])
    stay_ids = table['stay_id'].to_numpy().astype(int)
    # times = table['stay_time'].to_numpy().astype(float)
    diffs = np.diff(stay_ids)
    positions = np.where(diffs < 0)[0]
    # print(positions)
    if len(positions) > 0:
        print(stay_ids[positions[0] - 2:positions[0]+3])
        print(diffs[positions[0] - 2:positions[0]+3])
        unique, indices = np.unique(stay_ids, return_index=True)
        print(unique[np.argsort(indices)])
    assert not np.any(diffs < 0)


@pytest.mark.parametrize(
    'dataset_path',
    [
        'datasets/mimic_demo/data/parquet/features',
        'datasets/physionet2019/data/parquet/features',
        'datasets/mimic/data/parquet/features',
        'datasets/hirid/data/parquet/features',
        'datasets/eicu/data/parquet/features',
        'datasets/aumc/data/parquet/features'
    ]
)
def test_sorted_ids_and_times_features(dataset_path):
    table = pq.read_table(dataset_path, columns=['stay_id', 'stay_time'])
    stay_ids = table['stay_id'].to_numpy().astype(int)
    diffs = np.diff(stay_ids)
    positions = np.where(diffs < 0)[0]
    # print(positions)
    if len(positions) > 0:
        print(stay_ids[positions[0] - 2:positions[0]+3])
        print(diffs[positions[0] - 2:positions[0]+3])
        unique, indices = np.unique(stay_ids, return_index=True)
        print(unique[np.argsort(indices)])
    assert not np.any(diffs < 0)
    # times = table['stay_time'].to_numpy().astype(float)
    # unique_stays, inverse, counts = np.unique(stay_ids, return_inverse=True, return_counts=True)
    # time_offsets = np.roll(np.cumsum(counts), 1)
    # time_offsets[0] = 0
    # time_diffs = np.diff(times + time_offsets[inverse])
    # assert not np.any(time_diffs < 0)


