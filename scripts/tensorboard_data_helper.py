import re
import numpy as np
from collections import defaultdict
import pandas as pd
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer


class TensorboardDataHelper():
    """Class to help extrat summary values from the end of multiple runs"""
    def __init__(self, logdir, tags=None, tag_filter_fn=None, n_values=1, run_filter_fn=lambda a: True, keep_nans=False):
        if tags is None and tag_filter_fn is None:
            raise ValueError('Either tags or tag_filter_fn must be defined!')
        if tags is not None and tag_filter_fn is not None:
            raise ValueError('Only one of tags or tag_filter_fn can be defined at once!')

        self.logdir = logdir
        self.n_values = n_values
        self.run_filter_fn = run_filter_fn
        self.keep_nans = keep_nans

        if tag_filter_fn is not None:
            self.tag_filter_fn = tag_filter_fn
        else:
            self.selected_tags = tags
            self.tag_filter_fn = lambda a: a in self.selected_tags

        self.event_multiplexer = EventMultiplexer().AddRunsFromDirectory(logdir)
        self.reload()

    def reload(self):
        self.event_multiplexer.Reload()
        self.runs_and_scalar_tags = {run: values['scalars']
                                     for run, values in self.event_multiplexer.Runs().items()
                                     if self.run_filter_fn(run)
                                     }

    def get_matching_runs(self):
        return self.runs_and_scalar_tags.keys()

    def _get_last_events(self, list_of_events):
        """Get last scalars in terms of training step"""
        return list_of_events
        def get_training_step(event):
            return event.step
        events = sorted(list_of_events, key=get_training_step)
        if not self.keep_nans:
            events = filter(lambda ev: np.isfinite(ev.value), events)
        return events[-self.n_values:]

    def generate_directory_of_values(self):
        # Default value of new key is a new dictionary
        # makes code look nicer :)
        output = defaultdict(dict)
        for run, tags in self.runs_and_scalar_tags.items():
            for tag in tags:
                # skip unwanted tags
                if not self.tag_filter_fn(tag):
                    continue

                all_scalars = self.event_multiplexer.Scalars(run, tag)
                for ev in self._get_last_events(all_scalars):
                    output[(run, ev.step)][tag] = ev.value
        return output

    def generate_pandas_dataframe(self):
        dict_of_values = self.generate_directory_of_values()
        df = pd.DataFrame.from_dict(dict_of_values, orient='index')
        df.index.names = ['run', 'step']
        return df


def main(tf_logdir, run_regex, tag_regex, n_values, outputfile):
    def tag_selector(tag):
        return re.search(tag_regex, tag) is not None

    def run_selector(run):
        return re.search(run_regex, run) is not None

    print('Saving runs in {tf_logdir} matching {run_regex} and data of tags matching {tag_regex} to {outputfile}'.format(**locals()))
    tfboard_helper = TensorboardDataHelper(tf_logdir, tag_filter_fn=tag_selector, run_filter_fn=run_selector, n_values=n_values)

    print('Matching runs: {}'.format(list(tfboard_helper.get_matching_runs())))
    df = tfboard_helper.generate_pandas_dataframe()
    df.to_csv(outputfile)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read most recent data from tensorboard log and write to csv file')
    parser.add_argument('tf_logdir', help='Log directory of tensorboard')
    parser.add_argument('outputfile', help='Path where the resulting csv should be stored')
    parser.add_argument('--filter-runs', help='Regex used to filter runs (default: "*")', default='.*')
    parser.add_argument('--filter-tags', help='Regex used to filter tags (default: "*")', default='.*')
    parser.add_argument('--last', help='Store last # non-nan values of each run (default: 1)', type=int, default=1)

    args = parser.parse_args()
    main(args.tf_logdir, args.filter_runs, args.filter_tags, args.last, args.outputfile)
