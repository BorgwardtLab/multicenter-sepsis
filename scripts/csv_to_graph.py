"""Create `networkx` graph from CSV of results."""

import networkx as nx
import pandas as pd

import argparse
import itertools
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument(
        '-o', '--output',
        default='/tmp/graph.graphml',
        type=str,
        help='Output file'
    )
    parser.add_argument(
        '-m', '--metric',
        type=str,
        default='auc_mean',
        help='Metric to use for edge weights'
    )

    args = parser.parse_args()

    df = pd.read_csv(args.INPUT)

    dataset_names = df['train_dataset'].unique().tolist()
    dataset_names.extend(df['eval_dataset'].unique().tolist())

    dataset_names = list(set(dataset_names))

    G = nx.DiGraph()
    G.add_nodes_from(dataset_names)

    for source, target in itertools.permutations(dataset_names, r=2):

        # No self-edges, please!
        if source == target:
            continue

        # Let's check whether we have the measurement in the file. If
        # not, we just ignore it.
        df_ = df.query(
            'train_dataset == @source and eval_dataset == @target'
        )

        row = df_.iloc[0]
        weight = row[args.metric]

        G.add_edge(source, target, weight=weight)

    nx.write_graphml(G, args.output)
