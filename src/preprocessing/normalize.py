"""Normalize input data set in parallel."""

import argparse
import json
import pathlib

import dask.dataframe as dd
import numpy as np

from src.variables.mapping import VariableMapping


# TODO: use environment variable here?
VM_CONFIG_PATH = str(
    pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config/variables.json'
    )
)

VM_DEFAULT = VariableMapping(VM_CONFIG_PATH)


# # read patient ids from split files
# patient_ids = 
# # read columns to drop
# # do some stuff with the variable map
# drop_cols = 
# raw_data = dd.read_parquet(
#     input_filename,
#     columns=VM_DEFAULT.core_set,
#     engine="pyarrow-dataset",
#     chunksize=1,
# )
# norm = Normalizer(patient_ids, drop_cols=[])
# norm = norm.fit(raw_data)
# means, stds = dask.compute(norm.stats['means'], norm.stats['stds'])
# # Write to json file


def get_patient_ids_by_split(dataset, split_info, split_name):
    """Get patient IDs by split name.

    Parameters
    ----------
    dataset : str
        Data set to get split information for, such as 'demo'.

    split_info : dict
        Information about all potential splits for the given data set.
        This is typically a JSON file.

    split_name : str
        Name of split, such as 'dev'.

    Returns
    -------
    Array of patient IDs in the desired split.
    """
    if split_name == 'dev':
        ids = split_info[dataset][split_name]['total']
    else:
        # TODO: this only provides `dev` data for now.
        prefix, count = split_name.split('_')
        ids = split_info[dataset]['dev'][f'split_{count}'][prefix]

    return np.asarray(ids)


def main(input_filename, split_filename, output_filename):
    """Perform normalization of input data, subject to a certain split."""
    with open(split_filename) as f:
        split_info = json.load(f)

    # TODO: these variables need to come from somewhere else
    dataset = 'demo'
    split_name = 'dev'

    patient_ids = get_patient_ids_by_split(dataset, split_info, split_name)

    raw_data = dd.read_parquet(
        input_filename,
        # FIXME: missing columns?
        #columns=VM_DEFAULT.core_set,
        engine="pyarrow-dataset",
        chunksize=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",                     # FIXME
        default='./dump/features.parquet',  # FIXME
        type=str,
        help="Path to parquet file or folder with parquet files containing "
             "the raw data.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        required=False,
        default='./dump/master_splits.json',  # FIXME
        help="JSON file containing split information. Required to ensure "
             "normalization is only computed using the dev split.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        help="Output file path to write parquet file with features.",
    )

    args = parser.parse_args()

    # TODO: perform additional sanity checks; do we want to overwrite
    # existing output files, for example?

    main(args.input_file, args.split_file, args.output_file)
