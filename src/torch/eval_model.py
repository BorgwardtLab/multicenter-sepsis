"""Routine for loading and evaluating a model."""
from argparse import ArgumentParser, Namespace
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from src.torch.torch_utils import variable_length_collate
import src.torch.models


def expand_time(instance, transform_each=lambda a: a,
                transform_once=lambda a: a):
    """Convert a single instance into multiple sliced instances.

    This prevents furture information of an instance from being accessed by the model.

    Args:
        instance: dict of values from an instance. Assumes each element is an
            array where the time dimension is first.
        transform_each: Function applied to each slice of the instance. This is
            usually dependent on the model.
        transform_once: Function applied to the whole instance prior to
            slicing. This is usually a normalization or does something with the
            labels.

    Returns:
        List of fake slice instances.
    """
    instance = transform_once(instance)
    length = len(instance['times'])
    instances_out = []

    for i in range(1, length+1):
        instances_out.append(
            transform_each(
                {key: value[:i] for key, value in instance.items()}))
    return instances_out


def variable_len_colate_unwrap(fake_batch):
    """Pytorch adds annother batch dimension around our fake instances.

    Remove that and pass result to variable length collate function.
    """
    return variable_length_collate(fake_batch[0])


def online_eval(model, dataset):
    # TODO: Continue here
    # You shall not...
    pass


def main(model, checkpoint, dataset, splits):
    """Main function to evaluate a model."""

    model_cls = getattr(src.torch.models, model)
    dataset_cls = getattr(src.datasets, dataset)
    model = model_cls.load_from_checkpoint(
        checkpoint,
        hparam_overrides={
            'dataset': dataset
        }
    )
    model.eval()
    results = {}
    for split in splits:
        results[split] = online_eval(
            model, dataset_cls(split=split), batch_size=32)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', choices=src.torch.models.__all__, type=str,
                        default='AttentionModel')
    parser.add_argument(
        '--dataset', required=True, type=str, choices=src.datasets.__all__,
    )
    parser.add_argument(
        '--splits', default=['validation'], choices=['validation', 'testing'],
        type=str, nargs='+')
    parser.add_argument('--checkpoint-path', required=True, type=str)
    parser.add_argument('--gpus', type=int, nargs='+', default=None)
    # figure out which model to use
    temp_args = parser.parse_args()

    # let the model add what it wants
    model_cls = getattr(src.torch.models, temp_args.model)

    params = parser.parse_args()
    main(params.model, params.checkpoint_path, params.dataset, params.splits)
