"""Feature extraction pipeline."""
import json
from pathlib import Path
import time

import dask.dataframe as dd
from dask.distributed import Client, progress
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.preprocessing.transforms import (
    ApplyOnNormalized,
    BoolToFloat,
    CalculateUtilityScores,
    DerivedFeatures,
    InvalidTimesFiltration,
    LookbackFeatures,
    MeasurementCounterandIndicators,
    Normalizer,
    SignatureFeatures,
    WaveletFeatures,
)
from src.variables.mapping import VariableMapping

# warnings.filterwarnings("error")
# dask.config.set(scheduler='single-threaded')

VM_CONFIG_PATH = str(
    Path(__file__).parent.parent.parent.joinpath("config/variables.json")
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


def read_dev_split_patients(splitfile):
    with open(splitfile, "r") as f:
        split_info = json.load(f)
    return split_info["dev"]["total"]


def sort_time(df):
    res = df.sort_values(VM_DEFAULT("time"), axis=0)
    return res


def check_time_sorted(df):
    return (df[VM_DEFAULT("time")].diff() <= 0).sum() == 0


# def get_rows_per_patient(filename):
#     ids = pq.read_table(filename, columns=[VM_DEFAULT("id")])
#     unique_ids, counts = np.unique(ids, return_counts=True)
#     return OrderedDict(zip(unique_ids, counts))
# 
# 
# def compute_devisions(rows_per_patient, max_rows):
#     divisions = []
#     cur_count = 0
#     for patient_id, n_obs in rows_per_patient.items():
#         if (len(divisions) == 0) or (cur_count + n_obs > max_rows and cur_count != 0):
#             # Either first partition or we cannot add more to the current
#             # partition
#             divisions.append(patient_id)
#             cur_count = n_obs
#         else:
#             cur_count += n_obs
# 
#     if cur_count != 0:
#         divisions.append(list(rows_per_patient.keys())[-1])
#     return np.array(divisions)

def main(input_filename, split_filename, output_filename, n_workers):
    client = Client(
        n_workers=n_workers,
        memory_limit="20GB",
        threads_per_worker=1,
        local_directory="/local0/tmp/dask2",
    )
    start = time.time()
    # print("Computing patient partitions...")
    # rows_per_patient = get_rows_per_patient(input_filename)
    # # Initial divisions
    # divisions1 = compute_devisions(rows_per_patient, max_partition_size)
    # # After lookback features
    # divisions2 = compute_devisions(rows_per_patient, max_partition_size // 5)
    # # After wavelets
    # divisions3 = compute_devisions(
    #     rows_per_patient, (max_partition_size // 10) // 2)
    # del rows_per_patient

    norm_ids = read_dev_split_patients(split_filename)
    data = dd.read_parquet(
        input_filename,
        columns=VM_DEFAULT.core_set, #[1:],
        #index=VM_DEFAULT("id"),
        engine='pyarrow',
        split_row_groups=5  # This assumes that there are approx 100 rows per row group
    )
    data = data.set_index(VM_DEFAULT("id"), sorted=True)
    data = data.groupby(VM_DEFAULT("id"), sort=False, group_keys=False) \
        .apply(sort_time)
    # is_sorted = raw_data.groupby(raw_data.index.name, group_keys=False).apply(check_time_sorted)
    # assert all(is_sorted.compute())
    # raw_data = raw_data.set_index(VM_DEFAULT("id"), sorted=True)

    data_pipeline = Pipeline(
        [
            ("bool_to_float", BoolToFloat()),
            ("derived_features", DerivedFeatures(VM_DEFAULT, suffix="locf")),
            (
                "lookback_features",
                LookbackFeatures(suffices=["_raw", "_derived"]),
            ),
            ("measurement_counts", MeasurementCounterandIndicators(suffix="_raw")),
            ("normalized_feature_transforms", ApplyOnNormalized(
                Normalizer(norm_ids, suffix=[
                           "_locf", "_derived"], assign_values=False),
                [
                    WaveletFeatures(suffix="_locf"),
                    SignatureFeatures(suffices=["_locf", "_derived"])
                ]
            )),
            (
                "calculate_target",
                CalculateUtilityScores(label=VM_DEFAULT("label")),
            ),
            (
                "filter_invalid_times",
                InvalidTimesFiltration(vm=VM_DEFAULT, suffix="_raw"),
            ),
        ]
    )
    data = data_pipeline.fit_transform(data)
    # TODO: There is still an issue when writing the metadata file. It seems
    # like the worker which should write out the metadata runs into memory
    # issues. All in all we could try to do this in a separate step or collect
    # the metadata in the main thread and write it after the cluster workers
    # are killed. Then we would for sure have enough memory.
    all_done = data.to_parquet(
        output_filename, append=False, overwrite=True,
        engine='pyarrow-dataset', write_metadata_file=False, compute=False,
        compression='SNAPPY', write_statistics=True,
        row_group_size=2000, use_dictionary=False #row_group_size=500
    )
    future = client.compute(all_done)
    progress(future)
    print()  # Don't overwrite the progressbar
    future.result()
    client.close()
    outputpath = Path(output_filename)
    print('Reading all metadata...')
    schema = None
    metadata = []
    for file in tqdm(outputpath.glob('*.parquet')):
        if schema is None:
            schema = pq.read_schema(file)
        metadata.append(pq.read_metadata(file))
    pq.write_metadata(
        schema, outputpath / '_metadata', metadata_collector=metadata)
    print("Preprocessing completed after {:.2f} seconds".format(
        time.time() - start))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_data",
        type=str,
        help="Path to parquet file or folder with parquet files containing the "
        "raw data.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Json file containing split information. Needed to ensure "
        "normalization is only computed using the dev split.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path to write parquet file with features.",
    )
    parser.add_argument(
        "--n-workers",
        default=30,
        type=int,
        help="Number of dask workers to start for parallel processing of data.",
    )
    args = parser.parse_args()
    assert Path(args.input_data).exists()
    assert Path(args.split_file).exists()
    assert Path(args.output).parent.exists()
    main(
        args.input_data,
        args.split_file,
        args.output,
        args.n_workers
    )
