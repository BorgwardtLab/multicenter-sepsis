"""Utility functions for pytorch."""
import math
import json
import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd

from src.evaluation.sklearn_utils import (nanany, NotOnsetLabelError)
from src.evaluation import shift_onset_label

class TbWithBestValueLogger(TensorBoardLogger):
    """Tensorboard logger which also tracks the best value of metrics."""

    @staticmethod
    def __convert_metric_names(metrics: dict) -> dict:
        def to_best(name):
            fragments = name.split('/')
            new_name = (
                'best_' + fragments[0] if len(fragments) == 1
                else fragments[0] + '/best_' + fragments[1]
            )
            return new_name

        return {
            to_best(name): val
            for name, val in metrics.items()
        }

    @staticmethod
    def __extract_direction(metrics: dict) -> dict:
        """Compute multiplicative factor for direction of improvement.

        If the initial value is +inf smaller values are considered better, thus
        a direction of -1 is returned.  If the initial value is -inf, larger
        values are considered better and thus a direction of 1 is returned.
        """
        return {
            name: -1 if val > 0 else 1
            for name, val in metrics.items()
        }

    def __init__(self, save_dir, initial_values, add_best=False, **kwargs):
        super().__init__(save_dir, **kwargs)
        self.logging_hparams = False
        self.hparams_saved = False
        self.initial_values = initial_values
        self.add_best = add_best
        self.last = {}
        if add_best:
            self.best = self.__convert_metric_names(initial_values)
            self.direction = self.__extract_direction(self.best)

    def log_best(self, metrics, step):
        """Take metrics and update the recorded best value."""
        metrics = self.__convert_metric_names(metrics)

        def is_best(metric, value):
            if metric not in self.best.keys():
                return False
            direction = self.direction[metric]
            if self.best[metric] * direction < value * direction:
                return True
            else:
                return False
        best_metrics = {
            name: value
            for name, value in metrics.items()
            if is_best(name, value)
        }
        self.best.update(best_metrics)
        super().log_metrics(best_metrics, step)

    @rank_zero_only
    def log_hyperparams(self, params):
        # Somehow hyperparameters are saved when a model is simply restored,
        # catch that here so we don't add an incorrect value when restoring.
        if self.hparams_saved:
            return
        # This is a not so nice hack, but required as the parent method calls
        # log_metrics, which would otherwise add the best metrics. On the first
        # call these are already present so we catch that.
        self.logging_hparams = True
        if self.add_best:
            super().log_hyperparams(
                params,
                {**self.initial_values, **self.best}
            )
        else:
            super().log_hyperparams(
                params,
                self.initial_values
            )
        self.logging_hparams = False
        self.hparams_saved = True

    @rank_zero_only
    def log_metrics(self, metrics, step):
        super().log_metrics(metrics, step)
        self.last.update(metrics)
        if self.add_best and not self.logging_hparams:
            self.log_best(metrics, step)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def variable_length_collate(batch):
    """Combine multiple instances of irregular lengths into a batch."""
    # This converts the batch from [{'a': 1, 'b': 2}, ..] format to
    # {'a': [1], 'b': [2]}
    transposed_items = {
        key: list(map(lambda a: a[key], batch))
        for key in batch[0].keys()
    }

    transposed_items['id'] = np.array(transposed_items['id'])

    lengths = np.array(list(map(len, transposed_items['labels'])))
    transposed_items['lengths'] = lengths
    max_len = max(lengths)

    # Pad labels with -100 as this is the mask value for nll_loss
    transposed_items['labels'] = np.stack(
        [
            np.pad(instance, ((0, max_len - len(instance)),),
                   mode='constant', constant_values=np.NaN)
            for instance in transposed_items['labels']
        ],
        axis=0
    )
    transposed_items['labels_shifted'] = np.stack(
        [
            np.pad(instance, ((0, max_len - len(instance)),),
                   mode='constant', constant_values=np.NaN)
            for instance in transposed_items['labels_shifted']
        ],
        axis=0
    )

    # transform `targets` similarly as labels:
    transposed_items['targets'] = np.stack(
        [
            np.pad(instance, ((0, max_len - len(instance)),),
                   mode='constant', constant_values=np.NaN)
            for instance in transposed_items['targets']
        ],
        axis=0
    )

    for key in ['times', 'ts', 'times_embedded']:
        if key not in transposed_items.keys():
            continue
        dims = len(transposed_items[key][0].shape)
        if dims == 1:
            padded_instances = [
                np.pad(instance, ((0, max_len - len(instance)),),
                       mode='constant')
                for instance in transposed_items[key]
            ]
        elif dims == 2:
            padded_instances = [
                np.pad(instance, ((0, max_len - len(instance)), (0, 0)),
                       mode='constant')
                for instance in transposed_items[key]
            ]
        else:
            raise ValueError(
                f'Unexpected dimensionality of instance data: {dims}')

        transposed_items[key] = np.stack(padded_instances, axis=0)

    transposed_items['statics'] = np.stack(transposed_items['statics'], axis=0)
    transposed_items['statics'] = \
        transposed_items['statics'].astype(np.float32)
    transposed_items['times'] = transposed_items['times'].astype(np.float32)
    transposed_items['ts'] = transposed_items['ts'].astype(np.float32)
    transposed_items['lengths'] = transposed_items['lengths'].astype(np.long)
    transposed_items['labels'] = transposed_items['labels'].astype(np.float32)
    transposed_items['labels_shifted'] = transposed_items['labels_shifted'].astype(np.float32)
    transposed_items['targets'] = transposed_items['targets'].astype(np.float32)

    return {
        key: torch.from_numpy(data) for key, data in
        transposed_items.items()
    }


# pylint: disable=R0903
class PositionalEncoding():
    """Apply positional encoding to instances."""

    def __init__(self, min_timescale, max_timescale, n_channels,
                 positions_key='times'):
        """PositionalEncoding.

        Args:
            min_timescale: minimal scale of values
            max_timescale: maximal scale of values
            n_channels: number of channels to use to encode position
        """
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.n_channels = n_channels
        self.positions_key = positions_key

        self._num_timescales = self.n_channels // 2
        self._inv_timescales = self._compute_inv_timescales()

    def _compute_inv_timescales(self):
        log_timescale_increment = (
            math.log(float(self.max_timescale) / float(self.min_timescale))
            / (float(self._num_timescales) - 1)
        )
        inv_timescales = (
            self.min_timescale
            * np.exp(
                np.arange(self._num_timescales)
                * -log_timescale_increment
            )
        )
        return inv_timescales

    def __call__(self, instance):
        """Apply positional encoding to instances."""
        # instance = instance.copy()  # We only want a shallow copy
        positions = instance[self.positions_key]
        scaled_time = (
            positions[:, np.newaxis] *
            self._inv_timescales[np.newaxis, :]
        )
        signal = np.concatenate(
            (np.sin(scaled_time), np.cos(scaled_time)),
            axis=1
        )
        positional_encoding = np.reshape(signal, (-1, self.n_channels))
        instance[self.positions_key+'_embedded'] = positional_encoding
        return instance


def to_observation_tuples(instance_dict):
    """Convert time series to tuple representation.

    Basically replace all NaNs in the ts field with zeros, add a measurement
    indicator vector and combine both with the time field.
    """
    instance_dict = instance_dict.copy()  # We only want a shallow copy
    if 'times_embedded' in instance_dict.keys():
        time = instance_dict['times_embedded']
    else:
        time = instance_dict['times']

    if len(time.shape) != 2:
        time = time[:, np.newaxis]

    ts_data = instance_dict['ts']
    # Inspired by "Why not to use Zero Imputation"
    # https://arxiv.org/abs/1906.00150
    # We augment "absence indicators", which should reduce distribution shift
    # and bias induced by measurements with low number of observations.
    invalid_measurements = ~np.isfinite(ts_data)
    ts_data = np.nan_to_num(ts_data)  # Replace NaNs with zero

    # Combine into a vector
    combined = np.concatenate((time, ts_data, invalid_measurements), axis=-1)

    # Replace time series data with new vectors
    instance_dict['ts'] = combined
    return instance_dict


def add_indicators(instance_dict):
    """Replace nans in input with 0 and add measurement indicators."""
    instance_dict = instance_dict.copy()  # We only want a shallow copy
    time = instance_dict['times']
    if len(time.shape) != 2:
        time = time[:, np.newaxis]

    ts_data = instance_dict['ts']
    invalid_measurements = ~np.isfinite(ts_data)
    ts_data = np.nan_to_num(ts_data)  # Replace NaNs with zero

    # Combine into a vector
    combined = np.concatenate((ts_data, invalid_measurements), axis=-1)

    # Replace time series data with new vectors
    instance_dict['ts'] = combined
    return instance_dict


def to_observation_tuples_without_indicators(instance_dict):
    """Convert time series to tuple representation.

    Basically replace all NaNs in the ts field with zeros and combine it with the time field.
    """
    instance_dict = instance_dict.copy()  # We only want a shallow copy
    if 'times_embedded' in instance_dict.keys():
        time = instance_dict['times_embedded']
    else:
        time = instance_dict['times']
    if len(time.shape) != 2:
        time = time[:, np.newaxis]

    ts_data = instance_dict['ts']
    ts_data = np.nan_to_num(ts_data)  # Replace NaNs with zero

    # Combine into a vector
    combined = np.concatenate((time, ts_data), axis=-1)

    # Replace time series data with new vectors
    instance_dict['ts'] = combined
    return instance_dict


class LabelPropagation():
    def __init__(self, shift_left, keys, shift_right=24):
        """ apply LabelPropagation
            - keys: list of keys that contain
              the shifted label
        """
        self.shift_left = shift_left
        self.shift_right = shift_right
        self.keys = keys

    def __call__(self, instance):
        label = instance['labels']
        label = pd.Series(label, index=instance['times'])
        new_label = shift_onset_label(
            instance['id'], 
            label, 
            self.shift_left,
            self.shift_right)
        for key in self.keys:
            instance[key] = new_label.values
        return instance


# pylint: disable=R0903
class ComposeTransformations():
    """Chain multiple transformations together."""

    def __init__(self, transformations):
        """ComposeTransformations.

        Args:
            transformations: List of transformations
        """
        self.transformations = transformations

    def __call__(self, instance):
        """Apply transformations to instance."""
        out = instance
        for transform in self.transformations:
            out = transform(out)
        return out
