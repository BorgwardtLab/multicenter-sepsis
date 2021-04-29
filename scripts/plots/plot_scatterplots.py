"""Plot set of model scatterplots.

This script draws evaluation curves for all models, supporting multiple
inputs. Only evaluation files are required.

Example call:

    python -m scripts.plots.plot_scatterplots \
        --output-directory /tmp/              \
        FILE

This will create plots in `tmp`.
"""

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from scripts.plots.plot_patient_eval import prev_dict

def interpolate_at(df, x):
    """Interpolate a data frame at certain positions.

    This is an auxiliary function for interpolating an indexed data
    frame at a certain position or at certain positions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame; must have index that is compatible with `x`.

    x : scalar or iterable
        Index value(s) to interpolate the data frame at. Must be
        compatible with the data type of the index.

    Returns
    -------
    Data frame evaluated at the specified index positions.
    """
    # Check whether object support iteration. If yes, we can build
    # a sequence index; if not, we have to convert the object into
    # something iterable.
    try:
        _ = (a for a in x)
        new_index = pd.Index(x)
    except TypeError:
        new_index = pd.Index([x])

    # Ensures that the data frame is sorted correctly based on its
    # index. We use `mergesort` in order to ensure stability. This
    # set of options will be reused later on.
    sort_options = {
        'ascending': False,
        'kind': 'mergesort',
    }
    df = df.sort_index(**sort_options)

    # TODO: have to decide whether to keep first index reaching the
    # desired level or last. The last has the advantage that it's a
    # more 'pessimistic' estimate since it will correspond to lower
    # thresholds.
    df = df[~df.index.duplicated(keep='last')]

    # Include the new index, sort again and then finally interpolate the
    # values.
    df = df.reindex(df.index.append(new_index).unique())
    df = df.sort_index(**sort_options)
    df = df.interpolate()

    return df.loc[new_index]


def get_coordinates(df, recall_threshold, level, x_stat='earliness_median'):
    """Get coordinate from model-based data frame."""
    recall_col = f'{level}_recall'
    precision_col = f'{level}_precision'

    df = df.set_index(recall_col)
    df = interpolate_at(df, recall_threshold)

    assert len(df) == 1, RuntimeError(
        f'Expected a single row, got {len(df)}.'
    )

    x = df[x_stat].values[0]
    y = df[precision_col].values[0]

    # TODO: find nicer names for these labels
    return x, x_stat, y, precision_col


def sanitise_model_name(model):
    """Return sanitised model name."""
    if model == 'AttentionModel':
        return 'attention model'
    else:
        return model


def make_scatterplot(
    df,
    ax,
    recall_threshold,
    level,
    point_alpha,
    line_alpha,
    prev,
):
    """Create model-based scatterplot from joint data frame."""
    # Will contain a single data frame to plot. This is slightly more
    # convenient because it permits us to use `seaborn` directly.
    plot_df = []

    for (model, repetition), df_ in df.groupby(['model', 'rep']):
        # We are tacitly assuming that all the labels remain the same
        # during the iteration.
        x, xlabel, y, ylabel = get_coordinates(df_, recall_threshold, level)

        plot_df.append(
            pd.DataFrame.from_dict({
                'x': [x],
                'y': [y],
                'model': [sanitise_model_name(model)],
                'repetition': [repetition],
            })
        )

    plot_df = pd.concat(plot_df)

    g = sns.scatterplot(
        x='x', y='y',
        data=plot_df,
        hue='model',
        ax=ax,
        alpha=point_alpha,
        linewidth=1.2,
        marker='x'
    )
    g.axhline(
        prev, 
        linestyle='--', color='black',
        linewidth=0.5
    )

    g.legend(loc='upper right', fontsize=7)
    g.set_ylabel(f'{ylabel} @ {recall_threshold:.2f}R')
    g.set_ylim((0.0, 1.0))
    g.set_xlabel(xlabel)

    # Summarise each model by one glyph, following the same colour map
    # that `seaborn` uses.
    palette = sns.color_palette()

    for index, (model, df_) in enumerate(plot_df.groupby('model')):
        x_mean = df_['x'].mean()
        y_mean = df_['y'].mean()
        x_sdev = df_['x'].std()
        y_sdev = df_['y'].std()

        g.vlines(
            x_mean,
            y_mean - y_sdev,
            y_mean + y_sdev,
            colors=palette[index],
            alpha=line_alpha,
            linewidth=1.2,
        )

        g.hlines(
            y_mean,
            x_mean - x_sdev,
            x_mean + x_sdev,
            colors=palette[index],
            alpha=line_alpha,
            linewidth=1.2,
        )

        g.scatter(
            x_mean, y_mean,
            color=palette[index],
            marker='o',
            alpha=line_alpha,
            linewidth=0.8,
            s=10
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'FILE',
        type=str,
        help='Input file. Must be a CSV containing information about all of '
             'the repetition runs of a model.'
    )
    parser.add_argument(
        '--output-directory',
        type=str,
        help='Output directory'
    )

    parser.add_argument(
        '--point-alpha',
        default=1.0,
        type=float,
        help='Alpha to use for all points'
    )

    parser.add_argument(
        '--line-alpha',
        default=1.0,
        type=float,
        help='Alpha to use for all lines and glyphs'
    )

    parser.add_argument(
        '-r', '--recall-threshold',
        default=0.8,
        type=float,
        help='Recall threshold in [0,1]'
    )

    parser.add_argument(
        '-l', '--level',
        default='pat',
        type=str,
        choices=['pat', 'tp'],
        help='Species patient or time point level'
    )

    parser.add_argument(
        '-s', '--show',
        action='store_true',
        help='If set, indicates that the resulting plots should be shown, '
             'not only saved to a file.'
    )

    args = parser.parse_args()

    df = pd.read_csv(args.FILE)

    for (source, target), df_ in df.groupby(['dataset_train', 'dataset_eval']):

        fig, ax = plt.subplots(figsize=(4, 4)) #6,4
        ax.set_box_aspect(1)
        
        # determine prevalence of eval dataset: 
        prev = prev_dict[target]['validation']['case_prevalence'] 
        
        plt.title(f'Train: {source}, Eval: {target}')

        make_scatterplot(
            df_,
            ax,
            args.recall_threshold,
            args.level,
            point_alpha=args.point_alpha,
            line_alpha=args.line_alpha,
            prev=prev
        )

        plt.tight_layout()

        if args.output_directory:

            os.makedirs(args.output_directory, exist_ok=True)

            filename = os.path.join(
                args.output_directory,
                f'scatterplot_{source}_{target}_'
                f'{args.level}_'
                f'thres_{100 * args.recall_threshold:.0f}.png'
            )

            print(f'Storing {filename}...')
            plt.savefig(filename, dpi=400)

        if args.show:
            plt.show()

        plt.close()
