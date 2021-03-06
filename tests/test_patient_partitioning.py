from collections import OrderedDict
import pandas as pd
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


])  # train and val use same code path
def test_datasets(dataset_file):
    rows_per_patient = get_rows_per_patient(dataset_file)
    pd_row_per_patient = pd.Series(
        index=rows_per_patient.keys(), data=rows_per_patient.values())
    divisions = compute_devisions(rows_per_patient, 1000)
    for begin, end in zip(divisions[:-1], divisions[1:]):
        assert pd_row_per_patient[begin:end].sum() <= 1000
