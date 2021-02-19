"""Feature extraction pipeline."""
import argparse
from pathlib import Path

import dask.dataframe as dd
from sklearn.pipeline import Pipeline

from src.variables.mapping import VariableMapping
from src.preprocessing.transforms import DerivedFeatures, MeasurementCounterandIndicators, Normalizer, DaskPersist, WaveletFeatures

from src.sklearn.data.transformers import LookbackFeatures

import ipdb
import warnings
# warnings.filterwarnings("error")

VM_CONFIG_PATH = \
    str(Path(__file__).parent.parent.parent.joinpath('config/variables.json'))

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


def convert_bool_to_float(ddf):
    bool_cols = [col for col in ddf.columns if ddf[col].dtype == bool]
    ddf[bool_cols] = ddf[bool_cols].astype(float)
    return ddf


def sort_time(df):
    res = df.sort_values(VM_DEFAULT('time'), axis=0)
    return res


def check_time_sorted(df):
    return (df[VM_DEFAULT('time')].diff() <= 0).sum() == 0


def main(input_filename, split_filename, output_filename):
    raw_data = dd.read_parquet(
        input_filename,
        columns=VM_DEFAULT.core_set,
        engine='pyarrow-dataset'
    )
    # Set id to be the index, then sort within each id to ensure correct time
    # ordering.
    raw_data = raw_data \
        .set_index(VM_DEFAULT('id'), sorted=False, nparitions='auto')
    raw_data = raw_data \
        .groupby(VM_DEFAULT('id'), group_keys=False) \
        .apply(sort_time, meta=raw_data) \
        .persist()
    # Just to be sure, check that time is sorted
    # sorted_check = raw_data \
    #     .groupby(VM_DEFAULT('id'), dropna=False) \
    #     .apply(check_time_sorted, meta=bool) \
    #     .compute()
    # assert sorted_check.all()
    raw_data = convert_bool_to_float(raw_data).persist()
    norm_ids = raw_data.index.head().to_list()
    data_pipeline = Pipeline([
        ('derived_features', DerivedFeatures(VM_DEFAULT, suffix='locf')),
        # ('persist_features', DaskPersist()),
        ('lookback_features', LookbackFeatures(
            vm=VM_DEFAULT, suffices=['_raw', '_derived'])),
        ('measurement_counts', MeasurementCounterandIndicators(suffix='_raw')),
        ('persist_pernorm', DaskPersist()),
        ('feature_normalizer', Normalizer(norm_ids,
                                          suffix=['_locf', '_derived'])),
        # ('persist_normalized', DaskPersist()),
        ('wavelet_features', WaveletFeatures(suffix='_locf', vm=VM_DEFAULT)),
        # ('signatures', SignatureFeatures(n_jobs=n_jobs,
        #                                  suffices=['_locf', '_derived'], concat_output=True)),  # n_jobs=2
        # ('calculate_target', CalculateUtilityScores(label=vm('label'))),
        # ('filter_invalid_times', InvalidTimesFiltration(vm=vm, suffix='_raw'))
    ])
    import time
    start = time.time()
    raw_data = data_pipeline.fit_transform(raw_data).compute()
    print(time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_data',
        type=str,
        help='Path to parquet file or folder with parquet files containing the '
        'raw data.'
    )
    parser.add_argument(
        '--split-file',
        type=str,
        # required=True,
        help='Json file containing split information. Needed to ensure '
        'normalization is only computed using the dev split.'
    )
    parser.add_argument(
        '--output',
        type=str,
        # required=True,
        help='Output file path to write parquet file with features.'
    )
    args = parser.parse_args()
    assert Path(args.input_data).exists()
    # assert os.path.exists(args.split_file)
    # assert os.path.exists(os.path.dirname(args.output))
    main(args.input_data, args.split_file, args.output)
