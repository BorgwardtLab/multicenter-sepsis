"""Routine for loading and evaluating a model."""
from argparse import ArgumentParser
from functools import partial
import json

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score)
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.evaluation import physionet2019_utility
from src.torch.torch_utils import (
    variable_length_collate, ComposeTransformations)
import src.torch.models


if torch.cuda.is_available():
    print('Running eval on GPU')
    device = 'cuda'
else:
    device = 'cpu'



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


def variable_len_collate_unwrap(fake_batch):
    """Pytorch adds annother batch dimension around our fake instances.

    Remove that and pass result to variable length collate function.
    """
    return variable_length_collate(fake_batch[0])


def scores_to_pred(fn):
    """Convert scores into predictions, assumes probabilities."""
    def wrapped(labels, scores):
        predictions = [(s >= 0.5).astype(float) for s in scores]
        return fn(labels, predictions)
    return wrapped


def concat(fn):
    """Concatenate predictions of instances to single vector."""
    def wrapped(labels, scores):
        return fn(np.concatenate(labels, 0), np.concatenate(scores, 0))
    return wrapped


def online_eval(model, dataset_cls, split, check_matching_unmasked=True):
    """Run online evaluation with future masking."""
    transforms = model.transforms
    # TODO: Make this more generic, if first transform is not label propagation
    transform_once = transforms[0]  # Usually LabelPropagation
    transforms = transforms[1:]
    transform_each = ComposeTransformations(transforms)

    # This function transforms each instance into a list of sliced instances,
    # where future information is removed.
    transform_fn = partial(
        expand_time,
        transform_each=transform_each,
        transform_once=transform_once
    )

    dataloader = DataLoader(
        dataset_cls(split=split, transform=transform_fn),
        # This passes one instance at a time to the collate function. It could
        # be that the expanded instance requires to much memory to be
        # processed in a single pass.
        batch_size=1,
        shuffle=False,
        collate_fn=variable_len_collate_unwrap,
        pin_memory=True
    )

    if check_matching_unmasked:
        dataloader_um = DataLoader(
            dataset_cls(
                split=split,
                transform=ComposeTransformations(model.transforms)
            ),
            batch_size=1,
            shuffle=False,
            collate_fn=variable_length_collate,
            pin_memory=True
        )

    labels = []
    predictions = []

    scores = {
        'auroc': concat(roc_auc_score),
        'average_precision': concat(average_precision_score),
        'balanced_accuracy': concat(scores_to_pred(balanced_accuracy_score)),
        'physionet2019_score':
            scores_to_pred(partial(
                physionet2019_utility,
                shift_labels=model.hparams.label_propagation
            ))
    }

    model.eval()
    with torch.no_grad():
        if check_matching_unmasked:
            for batch, batch_um in tqdm(
                    zip(dataloader, dataloader_um),
                    desc='Masked evaluation',
                    total=len(dataloader)):
                data, length, label = batch['ts'], batch['lengths'], batch['labels']
                data_um, length_um, label_um = (
                    batch_um['ts'], batch_um['lengths'], batch_um['labels'])
                last_index = length - 1
                batch_index = np.arange(len(label))
                output = model(data.to(device), length.to(device))
                output_um = model(data_um.to(device), length_um.to(device))
                labels.append(label[(batch_index, last_index)].numpy())
                pred = output[(batch_index, last_index)][:, 0]
                predictions.append(torch.sigmoid(pred).detach().cpu().numpy())
                assert np.allclose(pred.detach().cpu().numpy(), output_um[..., 0].detach().cpu().numpy(), atol=1e-6)
        else:
            for batch in tqdm(dataloader, desc='Masked evaluation', total=len(dataloader)):
                data, length, label = batch['ts'], batch['lengths'], batch['labels']
                last_index = length - 1
                batch_index = np.arange(len(label))
                output = model(data.to(device), length.to(device))
                labels.append(label[(batch_index, last_index)].numpy())
                pred = output[(batch_index, last_index)][:, 0]
                predictions.append(torch.sigmoid(pred).detach().cpu().numpy())

    # Compute scores
    output = {name: fn(labels, predictions) for name, fn in scores.items()}
    # Add predictions
    output['labels'] = [el.tolist() for el in labels]
    output['predictions'] = [el.tolist() for el in predictions]
    return output


def main(model, checkpoint, dataset, splits, output):
    """Main function to evaluate a model."""
    model_cls = getattr(src.torch.models, model)
    dataset_cls = getattr(src.datasets, dataset)
    model = model_cls.load_from_checkpoint(
        checkpoint,
        hparam_overrides={
            'dataset': dataset
        }
    )
    model.to(device)
    results = {}
    for split in splits:
        res = online_eval(model, dataset_cls, split)
        results.update(
            {"{}_{}".format(split, key): value for key, value in res.items()})

    print({
        key: value for key, value in results.items()
        if 'labels' not in key and 'predictions' not in key
    })

    if output is not None:
        with open(output, 'w') as f:
            json.dump(results, f)


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
    parser.add_argument('--output', type=str, default=None)
    params = parser.parse_args()

    main(
        params.model,
        params.checkpoint_path,
        params.dataset,
        params.splits,
        params.output
    )
