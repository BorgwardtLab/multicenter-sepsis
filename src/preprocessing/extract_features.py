"""Feature extraction pipeline."""
from src.preprocessing.transforms import (
    BoolToFloat,
    CalculateUtilityScores,
    DaskRepartition,
    DerivedFeatures,
    InvalidTimesFiltration,
    LookbackFeatures,
    MeasurementCounterandIndicators,
    Normalizer,
    PatientPartitioning,
    SignatureFeatures,
    WaveletFeatures,
)
from pathlib import Path
import time
import json
import gc

from dask import delayed
import dask.dataframe as dd
from dask.distributed import Client, as_completed
import pyarrow
import pyarrow.parquet as pq
import numpy as np
from sklearn.pipeline import Pipeline
from tqdm import tqdm
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


def compute_patient_partitions(filename, max_rows_per_partition):
    ids = pq.read_table(filename, columns=[VM_DEFAULT("id")])
    unique_ids, counts = np.unique(ids, return_counts=True)

    divisions = []
    cur_count = 0
    for patient_id, n_obs in zip(unique_ids, counts):
        if (len(divisions) == 0) or (cur_count + n_obs > max_rows_per_partition and cur_count != 0):
            # Either first partition or we cannot add more to the current
            # partition
            divisions.append(patient_id)
            cur_count = n_obs
        else:
            cur_count += n_obs

    if cur_count != 0:
        divisions.append(unique_ids[-1])
    del ids, unique_ids, counts
    return divisions


def main(
    input_filename, split_filename, output_filename, n_workers, max_partition_size
):
    client = Client(
        n_workers=n_workers,
        memory_limit="20GB",
        threads_per_worker=1,
        local_directory="/local0/tmp/dask2",
    )
    start = time.time()
    print("Computing patient partitions...")
    patient_partitions = compute_patient_partitions(
        input_filename, max_partition_size)
    norm_ids = read_dev_split_patients(split_filename)
    data = dd.read_parquet(
        input_filename,
        columns=VM_DEFAULT.core_set[1:],
        engine="pyarrow",
        index=VM_DEFAULT("id"),
        split_row_groups=10
    )
    # is_sorted = raw_data.groupby(raw_data.index.name, group_keys=False).apply(check_time_sorted)
    # assert all(is_sorted.compute())
    # raw_data = raw_data.set_index(VM_DEFAULT("id"), sorted=True)

    data_pipeline = Pipeline(
        [
            ("bool_to_float", BoolToFloat()),
            ("patient_partitions", DaskRepartition(divisions=patient_partitions)),
            ("derived_features", DerivedFeatures(VM_DEFAULT, suffix="locf")),
            (
                "lookback_features",
                LookbackFeatures(suffices=["_raw", "_derived"]),
            ),
            ("measurement_counts", MeasurementCounterandIndicators(suffix="_raw")),
            ("feature_normalizer", Normalizer(
                norm_ids, suffix=["_locf", "_derived"])),
            ("wavelet_features", WaveletFeatures(suffix="_locf")),
            (
                "signatures",
                SignatureFeatures(suffices=["_locf", "_derived"]),
            ),
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
    del data_pipeline

    # Move from dask dataframes to the delayed api, for conversion and writing
    # of partitions
    partitions = data.to_delayed(optimize_graph=True)
    n_partitions = len(partitions)
    del data
    gc.collect()

    # Initialize output file
    # Get future objects, and trigger computation
    output_file = None
    try:
        with tqdm(desc='Writing row groups', total=n_partitions) as pbar:
            for batch in as_completed(
                    client.compute(partitions), with_results=True).batches():
                for future, result in batch:
                    result = pyarrow.Table.from_pandas(result)
                    future.cancel()
                    del future

                    if output_file is None:
                        # Initialize writer here, as we now have access to the table
                        # schema
                        output_file = pq.ParquetWriter(
                            output_filename,
                            result.schema,
                            write_statistics=[VM_DEFAULT("id")],
                            compression='SNAPPY'
                        )
                    # Write partition as row group
                    output_file.write_table(result)
                    pbar.update(1)
    finally:
        # Close output file
        if output_file is not None:
            output_file.close()
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
    parser.add_argument(
        "--max-partition-size",
        default=1000,
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
        args.n_workers,
        args.max_partition_size,
    )
