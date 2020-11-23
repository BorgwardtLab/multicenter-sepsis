"""Aggregate results from runs."""
from argparse import ArgumentParser
from glob import glob
import json
import os

import pandas as pd
import yaml


def extract_run_details(run_dir):
    """Extract run details."""
    with open(os.path.join(run_dir, 'hparams.yaml'), 'r') as f:
        run_details = yaml.load(f, Loader=yaml.BaseLoader)
    return run_details


def extract_run_results(run_dir):
    """Extract run results."""
    with open(os.path.join(run_dir, 'result.json'), 'r') as f:
        results = json.load(f)
    # Flatten the results
    output = {}
    for split, metrics in results.items():
        for metric, val in metrics.items():
            if metric in ['labels', 'predictions']:
                continue
            output['{}_{}'.format(split, metric)] = val
    return output


def parse_run(run_dir):
    """Parse all details from a run."""
    run_details = extract_run_details(run_dir)
    run_details.update(extract_run_results(run_dir))
    run_details['run_dir'] = run_dir
    return run_details


def get_run_dirs(path):
    """Get all subfolders with successful runs (containing a result.json)."""
    run_dirs = [
        os.path.dirname(p)
        for p in glob(os.path.join(path, '**', 'result.json'), recursive=True)
    ]
    return run_dirs


def main(path, outputs, filter_columns):
    run_dirs = get_run_dirs(path)
    all_runs = [parse_run(d) for d in run_dirs]

    if filter_columns is not None:
        # Only keep columns in filter_columns
        all_runs = [
            {key: val for key, val in run.items() if val in filter_columns}
            for run in all_runs
        ]
    df = pd.DataFrame.from_records(all_runs)

    for output in outputs:
        if output.endswith('tex'):
            df.to_latex(output)
        elif output.endswith('csv'):
            df.to_csv(output)
        else:
            raise ValueError('Unknown output format for {}'.format(output))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--outputs', required=True, type=str, nargs='+')
    parser.add_argument('--filter_columns', default=None, type=str, nargs='+')
    args = parser.parse_args()

    main(args.path, args.outputs, args.filter_columns)
