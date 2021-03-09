from collections import OrderedDict
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pytest
from src.preprocessing.extract_features import get_rows_per_patient, compute_devisions


def test_simple_example():
    test_patients = {
        1: 100,
        2: 300,
        3: 600,
        4: 1500,
        5: 500,
        6: 500
    }
    divisions = compute_devisions(test_patients, 1000)
    assert divisions == (1, 4, 5, 6)


@pytest.mark.parametrize('dataset_file', [
    'datasets/downloads/mimic_demo_0.3.0.parquet',
    'datasets/downloads/mimic_0.3.0.parquet',
    'datasets/downloads/eicu_0.3.0.parquet',
    'datasets/downloads/hirid_0.3.0.parquet',
    'datasets/downloads/aumc_0.3.0.parquet',
])
def test_ids_sorted(dataset_file):
    ids = pq.read_table(dataset_file, columns=['stay_id'])
    assert np.all(np.diff(ids) >= 0)


@pytest.mark.parametrize('dataset_file,max_rows', [
    ('datasets/downloads/mimic_demo_0.3.0.parquet', 1000),
    ('datasets/downloads/mimic_0.3.0.parquet', 1000),
    ('datasets/downloads/eicu_0.3.0.parquet', 1000),
    ('datasets/downloads/hirid_0.3.0.parquet', 1000),
    ('datasets/downloads/aumc_0.3.0.parquet', 1000)
])
def test_datasets(dataset_file, max_rows):
    rows_per_patient = get_rows_per_patient(dataset_file)
    longest_id = max(rows_per_patient, key=rows_per_patient.get)
    print(f'Longest patient {longest_id} has {rows_per_patient[longest_id]} rows.') 
    pd_row_per_patient = pd.Series(
        index=rows_per_patient.keys(), data=rows_per_patient.values(),
        name='n_values')
    divisions = compute_devisions(rows_per_patient, max_rows)
    n_oversize = 0
    for begin, end in zip(divisions[:-1], divisions[1:]):
        # Pandas label based slicing is inclusive!
        if not pd_row_per_patient.loc[begin:end-1].sum() <= max_rows:
            print(pd_row_per_patient.loc[begin:end-1])
            n_oversize += 1
    assert n_oversize == 0
