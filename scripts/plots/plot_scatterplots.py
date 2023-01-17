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
from collections import defaultdict
import os

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from scripts.plots.plot_patient_eval import prev_dict
from scripts.plots.plot_roc import model_map, emory_map

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

def make_scatterplot(
    df,
    ax,
    recall_threshold,
    level,
    point_alpha,
    line_alpha,
    prev,
    aggregation='macro',
    data_names = None,
):
    """Create model-based scatterplot from joint data frame."""
    # Will contain a single data frame to plot. This is slightly more
    # convenient because it permits us to use `seaborn` directly.
    plot_df = []
    use_subsamples=True if 'subsample' in df.columns else False
    assert aggregation in ['micro','macro']
    if use_subsamples: 
        for (model, repetition, subsample), df_ in df.groupby(['model', 'rep', 'subsample']):
            # We are tacitly assuming that all the labels remain the same
            # during the iteration.
            x, xlabel, y, ylabel = get_coordinates(df_, recall_threshold, level)

            plot_df.append(
                pd.DataFrame.from_dict({
                    'x': [x],
                    'y': [y],
                    'model': [model_map(model)],
                    'repetition': [repetition],
                    'subsample': [subsample]
                })
            )
    else:
        for (model, repetition), df_ in df.groupby(['model', 'rep']):
            # We are tacitly assuming that all the labels remain the same
            # during the iteration.
            x, xlabel, y, ylabel = get_coordinates(df_, recall_threshold, level)

            plot_df.append(
                pd.DataFrame.from_dict({
                    'x': [x],
                    'y': [y],
                    'model': [model_map(model)],
                    'repetition': [repetition],
                })
            )

    plot_df = pd.concat(plot_df)
    # plot_df['x'] = plot_df['x'] * -1

    # if macro aggregation, average over subsamples:
    if use_subsamples & (aggregation == 'macro'):
        print('Averaging out the subsamples..') 
        plot_df = plot_df.groupby(['model','repetition']).mean()[['x','y']].reset_index()  
    
    if aggregation == 'micro':
        sns.set(color_codes=True)
        graph = sns.jointplot(
            data=plot_df,
            x='x', y='y', hue='model',
            ax=ax,
            kind="kde",
            alpha=point_alpha,
            legend=True
        )
        g = graph.ax_joint
        g.axhline(
            y=prev, 
            linestyle='--', color='black', 
            linewidth=0.5
        )
    else:
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
    ax = plt.gca()
    ax.invert_xaxis()
    ax.legend(loc='lower right', fontsize=7,  ncol=3)  #
    #g.legend(loc='upper right', fontsize=7)
 
    g.set_ylabel(f'Positive predictive value at {int(100*recall_threshold)}% Sensitivity')
    g.set_ylim((0.0, 0.75))
    g.set_xlabel('Median earliness (hours before onset)')

    # Summarise each model by one glyph, following the same colour map
    # that `seaborn` uses.
    palette = sns.color_palette()

    agg = defaultdict(list)

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

        # gather raw data for writing out:
        agg['model'].append(model)
        agg['x_mean'].append(x_mean)
        agg['x_std'].append(x_sdev)
        agg['y_mean'].append(y_mean)
        agg['y_std'].append(y_sdev)

    agg_df = pd.DataFrame(agg)
    for _df in [plot_df, agg_df]:
        for key, name in zip(['train_dataset','eval_dataset'], data_names):
            _df[key] = name
    return plot_df, agg_df

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
    parser.add_argument(
        '--aggregation',
        default='macro',
        type=str,
        choices=['macro', 'micro'],
        help="""How to aggregate over results. macro: subsamples are 
            averaged out, micro: bootstraps of the inner loop including 
            subsamples"""
    )
    
    args = parser.parse_args()

    aggregation =  args.aggregation

    df = pd.read_csv(args.FILE)
    for col in ['dataset_train','dataset_eval']:
        df[col] = df[col].apply(emory_map)
     
    plot_df_list = []
    agg_df_list = []
    for (source, target), df_ in df.groupby(['dataset_train', 'dataset_eval']):

        fig, ax = plt.subplots(figsize=(4, 4)) #6,4
        ax.set_box_aspect(1)
        
        # determine prevalence of eval dataset:
        if 'subsample' in df_.columns:
            prev = 0.188 #target prevalence (precomputed)
            use_subsamples = True
            prev_obs = df['subsampled_prevalence']
            prev_max = prev_obs.max()
            prev_min = prev_obs.min()
            # sanity check, as actually used prevalence could vary by tiny amount 
            assert prev*0.9 < prev_min
            assert prev*1.1 > prev_max
  
        else:
            # stratified splits, so val and test prevalence are identical
            assert len(df_['split'].unique()) == 1
            split = df_['split'].values[0]  
            prev = prev_dict[target][split]['case_prevalence']
            use_subsamples = False 
        
        plt.title(f'Trained on {source}, tested on {target}')

        plot_df_curr, agg_df_curr = make_scatterplot(
            df_,
            ax,
            args.recall_threshold,
            args.level,
            point_alpha=args.point_alpha,
            line_alpha=args.line_alpha,
            prev=prev,
            data_names = [source, target],
            aggregation = aggregation
        )
        plot_df_list.append(plot_df_curr)
        agg_df_list.append(agg_df_curr) 

        plt.tight_layout()

        if args.output_directory:

            os.makedirs(args.output_directory, exist_ok=True)

            filename = os.path.join(
                args.output_directory,
                f'scatterplot_{source}_{target}_'
                f'{args.level}_'
                f'thres_{100 * args.recall_threshold:.0f}'
            )
            if use_subsamples:
                filename += '_subsampled'
            filename += '.png'

            print(f'Storing {filename}...')
            plt.savefig(filename, dpi=400)

        if args.show:
            plt.show()

        plt.close()

    # write raw scatter data out:
    plot_df = pd.concat(plot_df_list)
    agg_df = pd.concat(agg_df_list)
    summary = agg_df.query("train_dataset == eval_dataset & train_dataset != 'physionet2019'").mean()
    print(f'Pat eval summary:', summary) 
    for df,name in zip([plot_df, agg_df], 
        ['scatter_raw_data', 'scatter_agg_data']):
        if aggregation != 'macro': #default
            name += f'_{aggregation}' 
        df.to_csv(
            os.path.join(args.output_directory,
                name + '.csv'
            ),
            index=False 
        )
    
