"""Script to load trained model and run shape analysis."""

import tempfile
import os
import torch
import sys
import wandb
import warnings

import numpy as np

from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler

from src.torch.eval_model import compute_md5hash, device
from src.torch.torch_utils import ComposeTransformations
import src.torch.datasets
import src.torch.models

import shap

wandb_api = wandb.Api()


# Shap evaluation perturbs the inputs, thus we cannot rely on the length
# argument.  Instead use Nan padding and reconstruct length in model.
def variable_length_collate_nan_padding(batch):
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
                   mode='constant', constant_values=-100)
            for instance in transposed_items['labels']
        ],
        axis=0
    )
    transposed_items['labels_shifted'] = np.stack(
        [
            np.pad(instance, ((0, max_len - len(instance)),),
                   mode='constant', constant_values=-100)
            for instance in transposed_items['labels_shifted']
        ],
        axis=0
    )

    # transform `targets` similarly as labels:
    transposed_items['targets'] = np.stack(
        [
            np.pad(instance, ((0, max_len - len(instance)),),
                   mode='constant', constant_values=-100)
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
                       mode='constant', constant_values=np.NaN)
                for instance in transposed_items[key]
            ]
        elif dims == 2:
            padded_instances = [
                np.pad(instance, ((0, max_len - len(instance)), (0, 0)),
                       mode='constant', constant_values=np.NaN)
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

    transposed_items['lengths'] = transposed_items['lengths'].astype(np.int32)

    transposed_items['labels'] = transposed_items['labels'].astype(np.float32)

    transposed_items['labels_shifted'] = \
        transposed_items['labels_shifted'].astype(np.float32)

    transposed_items['targets'] = \
        transposed_items['targets'].astype(np.float32)

    return {
        key: torch.from_numpy(data) for key, data in
        transposed_items.items()
    }


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


def get_model_and_dataset(run_id):
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

    return model, dataset


def get_feature_names(dataset):
    """Get names of features from dataset."""
    remove_columns = [
        dataset.TIME_COLUMN,
        dataset.LABEL_COLUMN,
        dataset.UTILITY_COLUMN
    ] + dataset.STATIC_COLUMNS
    return [col for col in dataset.columns if col not in remove_columns]


class ModelWrapper(torch.nn.Module):
    """Wrapper class for compatibility with shap analysis."""

    def __init__(self, model, hours_before_end=0):
        """Wrap a model to only return output at the end of the stay.

        Args:
            model: Model to wrap
            hours_before_end: Number of hours before the end of the stay at
                which we should extract the output.
        """
        super().__init__()
        self.model = model
        self.hours_before_end = hours_before_end

    def forward(self, x, lengths=None, statics=None):
        """Perform forward pass through model."""
        if lengths is None:
            # Assume we use nan to pad values. This helps when using shap for
            # explanations as it manipulates the input and automatically adds
            # noise to the lengths parameter (making it useless for us).
            not_all_nan = (~torch.all(torch.isnan(x), dim=-1)).long()
            # We want to find the last instance where not all inputs are nan.
            # We can do this by flipping the no_nan tensor along the time axis
            # and determining the position of the maximum. This should return
            # us the first maximum, i.e. the first time when (in the reversed
            # order) where the tensor does not contain only nans.
            # Strangely, torch.argmax and tensor.max do different things.

            # The release we suggest to use here is plagued by certain
            # issues that are actually to our advantage here: `argmax`
            # actually returns the last index, so there's no other ops
            # that we to perform. See [1] for more details.
            #
            # [1]: https://github.com/pytorch/pytorch/issues/47296
            if torch.__version__ == '1.6.0':
                lengths = not_all_nan.argmax(1) + 1

                warnings.warn(
                    'Using pure `argmax` function call to determine '
                    'lengths of data.'
                )

            # Use the proposed strategy from above: reverse the ordering
            # of time points, pick the maximum, and use its index. Here,
            # `argmax` would also work.
            else:
                lengths = not_all_nan.shape[1] - not_all_nan.flip(1)        \
                                                            .contiguous()   \
                                                            .max(1)         \
                                                            .indices

                warnings.warn(
                    'Using flipped `argmax`/`max` function call to determine '
                    'lengths of data. You should be using PyTorch 1.6.0.'
                )

            # Remove the nan values again prior to model input
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        out = self.model(x, lengths=lengths, statics=statics)
        index = lengths - self.hours_before_end - 1
        assert torch.all(index >= 0)
        return out[torch.arange(out.shape[0]), index]


def run_shap_analysis(
    model,
    dataset,
    hours_before_end=0,
    min_length=5,
    n_samples_data=50,
    n_samples_background=200,
):
    """Run shap analysis on a model dataset pair.

    Parameters
    ----------
    n_samples_data : int
        Number of samples to use from the data set. The data set will be
        subsampled.
    n_samples_background : int
        Number of samples to be used as the background for Shapley value
        calculations.
    """
    # Get instance with at least `min_length` datapoints.
    lengths = np.array(
        list(map(lambda instance: len(instance['labels']), dataset))
    )
    indices = np.where(lengths >= min_length)[0]
    subset = Subset(dataset, indices)
    sampler = iter(SubsetRandomSampler(range(len(subset))))

    data = [
        subset[i] for i in list(
            map(
                lambda x: next(sampler),
                range(min(n_samples_data, len(subset)))
            )
        )
    ]
    data = variable_length_collate_nan_padding(data)

    print('Data keys:', data.keys())

    def get_model_inputs(batch):
        return [batch['ts'].to(device)]

    sample_dataset = get_model_inputs(data)
    wrapped_model = ModelWrapper(model, hours_before_end=hours_before_end)
    explainer = shap.GradientExplainer(
        wrapped_model,
        sample_dataset,
    )

    shap_values = explainer.shap_values(
        [sample_dataset[0]], n_samples_background)

    out = {
        'shap_values': shap_values,
        'id': data['id'],
        'input': data['ts'],
        'lengths': data['lengths'],
        'labels': data['labels'],
        # 'times': first_batch['time'],
        'feature_names': get_feature_names(dataset)
    }

    return out


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('wandb_run', type=str)
    parser.add_argument(
        '-b', '--n-samples-background',
        type=int,
        default=200,
        help='Number of samples to use for the Shapley value computation.'
    )

    parser.add_argument(
        '-n', '--n-samples',
        type=int,
        default=50,
        help='Number of samples to include in data set for Shapley value '
             'computation.'
    )

    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='If set, overwrites results.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output path to store pickle with shap values.'
    )

    parser.add_argument('--hours_before_end', type=int, default=0, help="Number of hours prior to the end of stay to look at for feature importance estimation.")
    parser.add_argument('--min_length', type=int, default=5, help="Minimal length of instance in order to be used for background.")

    params = parser.parse_args()

    if os.path.exists(params.output) and not params.force:
        warnings.warn(
            f'Refusing to overwrite "{params.output}" without `--force`.'
        )
        sys.exit(-1)

    model, dataset = get_model_and_dataset(params.wandb_run)

    shap_values = run_shap_analysis(
        model, dataset,
        hours_before_end=params.hours_before_end,
        n_samples_data=params.n_samples,
        n_samples_background=params.n_samples_background,
        min_length=params.min_length,
    )

    import pickle
    with open(params.output, 'wb') as f:
        pickle.dump(shap_values, f)
