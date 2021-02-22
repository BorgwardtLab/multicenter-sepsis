"""Normalize input data set in parallel."""

import argparse
import dask
import json
import pathlib

import dask.dataframe as dd

from src.sklearn.loading import SplitInfo

from src.preprocessing.transforms import Normalizer

from src.variables.mapping import VariableMapping


# TODO: use environment variable here
# TODO: might want to support the index as well
VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


def main(
    input_filename,
    split_filename,
    output_filename,
    split_name,
):
    """Perform normalization of input data, subject to a certain split."""
    patient_ids = SplitInfo(split_filename)(split_name)

    raw_data = dd.read_parquet(
        input_filename,
        engine="pyarrow-dataset",
        chunksize=1,
    )

    drop_cols = [
        VM_DEFAULT('label'),
        VM_DEFAULT('sex'),
        VM_DEFAULT('time'),
    ]

    norm = Normalizer(patient_ids, drop_cols=drop_cols)
    norm = norm.fit(raw_data)
    means, stds = dask.compute(
        norm.stats['means'],
        norm.stats['stds']
    )

    results = {
        'means': means.to_dict(),
        'stds': stds.to_dict(),
    }

    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        required=True,
        type=str,
        help="Path to parquet file or folder with parquet files containing "
             "the raw data.",
    )

    parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="JSON file containing split information. Required to ensure "
             "normalization is only computed using the dev split.",
    )
    parser.add_argument(
        '--split-name',
        type=str,
        required=True,
        default='train',
        choices=['train', 'test', 'val', 'dev'],
        help='Indicate split for which the normalization shall be performed.'
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file path to write parquet file with features.",
    )

    args = parser.parse_args()

    # TODO: perform additional sanity checks; do we want to overwrite
    # existing output files, for example?

    main(
        args.input_file,
        args.split_file,
        args.output_file,
        args.split_name,
    )
