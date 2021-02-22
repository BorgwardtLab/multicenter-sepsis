import argparse
import pyarrow
import pyarrow.parquet as pq


def main(input_filename, output_filename, chunksize):
    input_data = pq.read_table(input_filename)
    index_col = input_data.column_names[0]
    with pq.ParquetWriter(
            output_filename, input_data.schema, write_statistics=[index_col]) as f:
        f.write_table(input_data, row_group_size=chunksize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--chunksize', type=int, default=500)
    args = parser.parse_args()

    main(args.input_filename, args.output, args.chunksize)
