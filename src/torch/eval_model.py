"""Routine for loading and evaluating a model."""
from argparse import ArgumentParser
from hashlib import md5
from functools import partial
import os
import yaml
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
                {
                    # ts values
                    **{key: value[:i] for key, value in instance.items()
                       if hasattr(value, '__len__') and key != 'statics'},
                    # non-ts values
                    **{key: value for key, value in instance.items()
                       if (not hasattr(value, '__len__')) or key == 'statics'},
                }
            ))
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


def online_eval(model, dataset_cls, split, check_matching_unmasked=False, device=device, **kwargs):
    """Run online evaluation with future masking."""
    model = model.to(device)
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
        dataset_cls(split=split, transform=transform_fn, **kwargs),
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
    scores = []
    predictions = []
    ids = []
    times = []

    model.eval()
    with torch.no_grad():
        if check_matching_unmasked:
            for batch, batch_um in tqdm(
                    zip(dataloader, dataloader_um),
                    desc='Masked evaluation',
                    total=len(dataloader)):
                time, data, length, label = (
                    batch['times'],
                    batch['ts'],
                    batch['lengths'],
                    batch['labels']
                )
                time_um, data_um, length_um, label_um = (
                    batch_um['times'],
                    batch_um['ts'],
                    batch_um['lengths'],
                    batch_um['labels']
                )
                last_index = length - 1
                batch_index = np.arange(len(label))
                # Apply model
                output = model(data.to(device), length.to(device))
                pred = output[(batch_index, last_index)][:, 0]
                output_um = model(data_um.to(device), length_um.to(device))
                labels.append(label[(batch_index, last_index)].numpy())
                times.append(time[(batch_index, last_index)].numpy())
                scores.append(torch.sigmoid(pred).cpu().numpy())
                predictions.append((scores[-1] >= 0.5).astype(int))
                ids.append(int(batch_um['id'].cpu().numpy()[0]))
                assert np.allclose(pred.cpu().numpy(
                ), output_um[..., 0].cpu().numpy(), atol=1e-6)
        else:
            for batch in tqdm(dataloader, desc='Masked evaluation', total=len(dataloader)):
                time, data, length, label = (
                    batch['times'],
                    batch['ts'],
                    batch['lengths'],
                    batch['labels']
                )
                last_index = length - 1
                batch_index = np.arange(len(label))
                output = model(data.to(device), length.to(device))
                labels.append(label[(batch_index, last_index)].numpy())
                times.append(time[(batch_index, last_index)]).numpy()
                pred = output[(batch_index, last_index)][:, 0]
                scores.append(torch.sigmoid(pred).cpu().numpy())
                predictions.append((scores[-1] >= 0.5).astype(int))
                ids.append(int(batch['id'][0].cpu().numpy()))
    scores_fns = {
        'auroc': concat(roc_auc_score),
        'average_precision': concat(average_precision_score),
        'balanced_accuracy': concat(scores_to_pred(balanced_accuracy_score)),
        'physionet2019_score':
            scores_to_pred(partial(
                physionet2019_utility,
                shift_labels=model.hparams.label_propagation
            ))
    }

    # Compute scores
    output = {name: fn(labels, scores) for name, fn in scores_fns.items()}
    # Add predictions
    output['labels'] = [el.tolist() for el in labels]
    output['scores'] = [el.tolist() for el in scores]
    output['predictions'] = [el.tolist() for el in predictions]
    output['times'] = [el.tolist() for el in times]
    output['ids'] = ids  # [el.tolist() for el in ids]
    return output


def compute_md5hash(filename, blocksize=1024):
    """Compute the md5 has of file with path filename."""
    filesize = os.path.getsize(filename)
    hash_md5 = md5()
    with open(filename, "rb") as f:
        iterator = iter(lambda: f.read(blocksize), b"")
        for chunk in tqdm(iterable=iterator, total=filesize // blocksize, unit='KB'):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_model_checkpoint_path(run_folder):
    """Get path to model checkpoint from run directory."""
    checkpoint_dir = os.path.join(run_folder, 'checkpoints')
    checkpoint_file = os.listdir(checkpoint_dir)[0]
    checkpoint = os.path.join(checkpoint_dir, checkpoint_file)
    return checkpoint


def extract_model_information(run_folder, checkpoint_path=None):
    with open(os.path.join(run_folder, 'hparams.yaml'), 'r') as f:
        run_info = yaml.load(f, Loader=yaml.BaseLoader)

    if checkpoint_path is None:
        checkpoint_path = get_model_checkpoint_path(run_folder)
    model_checksum = compute_md5hash(checkpoint_path)

    return {
        "model": run_info['model'],
        "model_path": checkpoint_path,
        "model_checksum": model_checksum,
        "model_params": run_info,
        "dataset_train": run_info['dataset']
    }


def main(run_folder, dataset, split, checkpoint_path, output):
    """Main function to evaluate a model."""
    out = extract_model_information(run_folder, checkpoint_path)
    out['dataset_eval'] = dataset
    out['split'] = split

    model_cls = getattr(src.torch.models, out['model'])
    dataset_cls = getattr(src.datasets, dataset)
    model = model_cls.load_from_checkpoint(
        out['model_path'],
        dataset=dataset
    )
    model.to(device)
    out.update(online_eval(model, dataset_cls, split))

    print({
        key: value for key, value in out.items()
        if key not in ['labels', 'predictions', 'scores', 'ids', 'times']
    })

    if output is not None:
        with open(output, 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run_folder', type=str, required=True)
    parser.add_argument(
        '--dataset', required=True, type=str, choices=src.datasets.__all__,
    )
    parser.add_argument(
        '--split',
        default='validation',
        choices=['train', 'validation', 'test'],
        type=str
    )
    parser.add_argument(
        '--checkpoint-path', required=False, default=None, type=str)
    parser.add_argument('--output', type=str, default=None)
    params = parser.parse_args()

    main(
        params.run_folder,
        params.dataset,
        params.split,
        params.checkpoint_path,
        params.output,
    )
