"""Plot ranking of variables or features."""


import argparse
import functools
import os

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'INPUT',
        nargs='+',
        type=str,
        help='Input file(s)'
    )

    args = parser.parse_args()

    data = [
        pd.read_csv(filename) for filename in args.INPUT
    ]

    annotations = []

    # Get 'hours' from the filename. Probably the dumbest parsing code
    # I ever wrote, but it works!
    for filename in args.INPUT:
        basename = os.path.basename(filename)

        hours = basename.split('_')[1]
        if not hours.endswith('h'):
            hours = 'all'

        if len(hours) == 2:
            hours = ' ' + hours

        annotations.append(hours)

    data = [
        df.rename(
            columns={
                'rank': hours
            }
        )
        for df, hours in zip(data, annotations)
    ]

    df = functools.reduce(
        lambda left, right: pd.merge(left, right, on='feature'), data
    )

    df = df.set_index('feature')
    df = df.sort_index(axis='columns')

    top_features = df.mean(axis='columns').sort_values().index[:20]
    df = df.loc[top_features]

    df = df.transpose()
    df.index.name = 'hours'

    df = df.reset_index().melt(
        'hours',
        var_name='feature', value_name='rank'
    )

    g = sns.catplot(
        x='hours',
        y='rank',
        hue='feature',
        data=df,
        kind='point',
        palette='husl'
    )

    plt.show()
