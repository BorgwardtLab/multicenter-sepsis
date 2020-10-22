"""Utility functions for pytorch."""
import math
import json
import numpy as np
import torch


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
    for key in ['times', 'ts']:
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

    # transposed_items['statics'] = np.stack(transposed_items['statics'], axis=0)
    # transposed_items['statics'] = \
    #     transposed_items['statics'].astype(np.float32)
    transposed_items['times'] = transposed_items['times'].astype(np.float32)
    transposed_items['ts'] = transposed_items['ts'].astype(np.float32)
    transposed_items['lengths'] = transposed_items['lengths'].astype(np.long)
    transposed_items['labels'] = transposed_items['labels'].astype(np.float32)

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
        instance = instance.copy()  # We only want a shallow copy
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
        instance[self.positions_key] = positional_encoding
        return instance


def to_observation_tuples(instance_dict):
    """Convert time series to tuple representation.

    Basically replace all NaNs in the ts field with zeros, add a measurement
    indicator vector and combine both with the time field.
    """
    instance_dict = instance_dict.copy()  # We only want a shallow copy
    time = instance_dict['times']
    if len(time.shape) != 2:
        time = time[:, np.newaxis]

    ts_data = instance_dict['ts']
    # sanity check, in case there should be any remaining NaNs (but there shouldn't) 
    ts_data = np.nan_to_num(ts_data)  # Replace NaNs with zero

    # Combine into a vector
    combined = np.concatenate((time, ts_data), axis=-1)
    
    # Replace time series data with new vectors
    instance_dict['ts'] = combined
    return instance_dict


class LabelPropagation():
    def __init__(self, hours_shift):
        self.hours_shift = hours_shift

    def __call__(self, instance):
        label = instance['labels']
        is_case = np.any(label)
        assert not np.any(np.isnan(label))
        if is_case:
            onset = np.argmax(label)
            # Check if label is a onset
            if not np.all(label[onset:]):
                raise ValueError('Did not get an onset label.')

            new_onset = onset + self.hours_shift
            new_onset = min(max(0, new_onset), len(label))
            new_onset_segment = np.ones(len(label) - new_onset)
            # NaNs should stay NaNs
            new_label = np.concatenate(
                [label[:new_onset], new_onset_segment], axis=0)
            instance['labels'] = new_label
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
