"""Utility functions for pytorch."""
import numpy as np
import torch


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
                   mode='constant', constant_values=-100)
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

    transposed_items['statics'] = np.stack(transposed_items['statics'], axis=0)

    transposed_items['statics'] = \
        transposed_items['statics'].astype(np.float32)
    transposed_items['times'] = transposed_items['times'].astype(np.float32)
    transposed_items['ts'] = transposed_items['ts'].astype(np.float32)
    transposed_items['lengths'] = transposed_items['lengths'].astype(np.long)
    transposed_items['labels'] = transposed_items['labels'].astype(np.long)

    return {
        key: torch.from_numpy(data) for key, data in
        transposed_items.items()
    }
