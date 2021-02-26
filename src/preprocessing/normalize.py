"""Normalize input data set in parallel."""

import argparse
import dask
import json
import pathlib
import os
import glob 
import dask.dataframe as ddf
import dask.diagnostics as dd

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
    split_name,
    repetition,
    output_filename,
):
    """Perform normalization of input data, subject to a certain split."""
    patient_ids = SplitInfo(split_filename)(split_name, repetition)

    progress_bar = dd.ProgressBar()
    progress_bar.register()
    if os.path.isdir(input_filename):
       input_filename = glob.glob(
            os.path.join(input_filename, '*.parquet' )
        ) 
    raw_data = ddf.read_parquet(
        input_filename,
        engine="pyarrow-legacy",
        chunksize=1
    )
    ind_cols = [col for col in raw_data.columns if '_indicator' in col]

    drop_cols = [
        VM_DEFAULT('label'),
        VM_DEFAULT('sex'),
        VM_DEFAULT('time'),
        VM_DEFAULT('utility'),
        *VM_DEFAULT.all_cat('baseline'),
        *ind_cols
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
             "normalization is only computed using the correct training split.",
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
        '-r', '--repetition',
        type=int,
        default=0,
        help='Repetition of split to load'
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file path.",
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='If set, overwrites output files'
    )

    args = parser.parse_args()

    for p in [args.input_file, args.split_file]:
        assert pathlib.Path(p).exists(), \
            RuntimeError(f'Path {p} does not exist')

    if pathlib.Path(args.output_file).exists() and not args.force:
        raise RuntimeError(f'Refusing to overwrite {args.output_file} unless '
                           f'`--force` is set.')

    main(
        args.input_file,
        args.split_file,
        args.split_name,
        args.repetition,
        args.output_file,
    )
