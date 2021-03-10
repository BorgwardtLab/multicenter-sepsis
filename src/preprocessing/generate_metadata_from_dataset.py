"""Gather metadata from individual parquet files and write to metadata file."""
import argparse
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm


def get_partition_id(file: Path):
    """Split off the part of the path that corresponds to the partition."""
    return int(file.stem.split('.')[-1])


def write_metadata_to_dataset(dataset_folder):
    """Read files from a dask generated dataset and write _metadata."""
    dataset_folder = Path(dataset_folder)
    schema = None
    metadata = []
    for file in tqdm(
            sorted(dataset_folder.glob('*.parquet'), key=get_partition_id),
            desc='Reading metadata'):
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
    write_metadata_to_dataset(args.dataset_folder)
