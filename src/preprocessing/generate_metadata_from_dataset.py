"""Gather metadata from individual parquet files and write to metadata file."""
import argparse
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm


def main(dataset_folder):
    dataset_folder = Path(dataset_folder)
    schema = None
    metadata = []
    for file in tqdm(dataset_folder.glob('*.parquet'), desc='Reading metadata'):
        if schema is None:
            schema = pq.read_schema(file)
        cur_metadata = pq.read_metadata(file)
        cur_metadata.set_file_path(str(file.relative_to(dataset_folder)))
        metadata.append(cur_metadata)
    pq.write_metadata(
        schema, dataset_folder / '_metadata', metadata_collector=metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str)
    args = parser.parse_args()
    main(args.dataset_folder)
