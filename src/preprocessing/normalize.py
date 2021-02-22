"""Normalize input data set in parallel."""

import argparse

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


def main(input_filename, split_filename, output_filename):
    """Perform normalization of input data, subject to a certain split."""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_data",
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
        "--output",
        type=str,
        required=True,
        help="Output file path to write parquet file with features.",
    )

    args = parser.parse_args()
