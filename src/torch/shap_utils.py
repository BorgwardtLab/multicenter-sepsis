"""Utility functions for interacting with Shapley values."""

import os
import tempfile
import wandb

import numpy as np

from src.torch.eval_model import compute_md5hash
from src.torch.eval_model import device

from src.torch.torch_utils import ComposeTransformations

import src.torch.models

wandb_api = wandb.Api()


def get_run_id(filename):
    """Extract run ID from filename."""
    run_id = os.path.basename(filename)
    run_id = os.path.splitext(run_id)[0]
    run_id = run_id[:8]
    run_id = f'sepsis/mc-sepsis/runs/{run_id}'
    return run_id


def extract_model_information(run_path, tmp):
    """Get model information from wandb run."""
    run = wandb_api.run(run_path)
    run_info = run.config
    checkpoint_path = None
    for f in run.files():
        if f.name.endswith('.ckpt'):
            file_desc = f.download(tmp)
            checkpoint_path = file_desc.name
            file_desc.close()
    if checkpoint_path is None:
        raise RuntimeError(
            f'Run "{run_path}" does not have a stored checkpoint file.')

    model_checksum = compute_md5hash(checkpoint_path)
    dataset_kwargs = {}
    for key in run_info.keys():
        if 'dataset_kwargs' in key:
            new_key = key.split('/')[-1]
            dataset_kwargs[new_key] = run_info[key]
    return run, {
        "model": run_info['model'],
        "run_id": run_path,
        "model_path": checkpoint_path,
        "model_checksum": model_checksum,
        "model_params": run_info,
        "dataset_train": run_info['dataset'],
        "task": run_info['task'],
        "label_propagation": run_info['label_propagation'],
        "rep": run_info['rep'],
        "dataset_kwargs": dataset_kwargs
    }


def get_model_and_dataset(run_id, return_out=False):
    """Get model and dataset from finished wandb run."""
    with tempfile.TemporaryDirectory() as tmp:
        # Download checkpoint to temporary directory
        run, out = extract_model_information(run_id, tmp)

        model_cls = getattr(src.torch.models, out['model'])
        model = model_cls.load_from_checkpoint(
            out['model_path'],
            dataset=out['dataset_train']
        )
    model.to(device)
    dataset = model.dataset_cls(
        split='test',
        transform=ComposeTransformations(model.transforms),
        **out['dataset_kwargs']
    )

    if return_out:
        return model, dataset, out
    else:
        return model, dataset


def pool(lengths, shapley_values, feature_values, hours_before=None):
    """Pool Shapley values and feature values.

    This function performs the pooling step required for Shapley values
    and feature values. Each time step of a time series will be treated
    as its own sample instance.

    Parameters
    ----------
    lengths:
        Vector of lengths.

    shapley_values:
        Shapley values.

    feature_values:
        Feature values.

    hours_before : int or `None`, optional
        If not `None`, only uses (at most) the last `hours_before`
        observations when reporting the Shapley values.

    Returns
    -------
    Tuple of pooled Shapely values and feature values.
    """
    # This is the straightforward way of doing it. Please don't judge,
    # or, if you do, don't judge too much :)
    shapley_values_pooled = []
    feature_values_pooled = []

    for index, (s_values, f_values) in enumerate(
        zip(shapley_values, feature_values)
    ):
        length = lengths[index]

        # Check whether we can remove everything but the last hours
        # before the maximum prediction score.
        if hours_before is not None:
            if length >= hours_before:
                s_values = s_values[length - hours_before:length, :]
                f_values = f_values[length - hours_before:length, :]

        # Just take *everything*.
        else:
            s_values = s_values[:length, :]
            f_values = f_values[:length, :]

        # Need to store mask for subsequent calculations. This is
        # required because we must only select features for which
        # the Shapley value is defined!
        mask = ~np.isnan(s_values).all(axis=1)

        s_values = s_values[mask, :]
        f_values = f_values[mask, :]

        shapley_values_pooled.append(s_values)
        feature_values_pooled.append(f_values)

    shapley_values_pooled = np.vstack(shapley_values_pooled)
    feature_values_pooled = np.vstack(feature_values_pooled)

    return shapley_values_pooled, feature_values_pooled


def get_aggregation_function(name):
    """Return aggregation function."""
    if name == 'absmax':
        def absmax(x):
            return max(x.min(), x.max(), key=abs)
        return absmax
    elif name == 'max':
        return np.max
    elif name == 'min':
        return np.min
    elif name == 'mean':
        return np.mean
    elif name == 'median':
        return np.median
