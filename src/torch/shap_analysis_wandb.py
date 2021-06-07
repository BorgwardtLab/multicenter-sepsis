"""Script to load trained model and run shape analysis."""
import json
import tempfile
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.torch.eval_model import compute_md5hash, device
from src.torch.torch_utils import ComposeTransformations, variable_length_collate
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

    transposed_items['lengths'] = transposed_items['lengths'].astype(np.long)

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
        split='validation',
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

            # TODO: It looks like something might still be wrong here!
            # FIXME: Here is an issue, please fix. Seems to only happen on GPU.
            print(not_all_nan.flip(1))
            print(not_all_nan.flip(1).contiguous().max(1).indices)
            lengths = not_all_nan.shape[1] - not_all_nan.flip(1).contiguous().max(1).indices
            print(lengths)

            # Remove the nan values again prior to model input
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        out = self.model(x, lengths=lengths, statics=statics)
        index = lengths - self.hours_before_end - 1
        assert torch.all(index >= 0)
        return out[torch.arange(out.shape[0]), index]


def run_shap_analysis(model, dataset, hours_before_end=0, n_samples=200, min_length=5, max_examples=50):
    """Run shap analysis on a model dataset pair."""
    # Get instance with at least min_length datapoints.
    lengths = np.array(list(map(lambda instance: len(instance['labels']), dataset)))
    indices = np.where(lengths >= min_length)[0]
    subset = Subset(dataset, indices)
    first_batch = [subset[i] for i in range(min(max_examples, len(subset)))]
    first_batch = variable_length_collate_nan_padding(first_batch)
    print(first_batch.keys())

    def get_model_inputs(batch):
        return [batch['ts'].to(device)]

    print(lengths[:min(max_examples, len(subset))])
    sample_dataset = get_model_inputs(first_batch)
    wrapped_model = ModelWrapper(model, hours_before_end=hours_before_end)
    explainer = shap.GradientExplainer(wrapped_model, sample_dataset, batch_size=50)
    n_examples = 2

    shap_values = explainer.shap_values(
        [sample_dataset[0][:n_examples]], n_samples)
    out = {
        'shap_values': shap_values,
        'input': first_batch['ts'][:n_examples],
        'lengths': first_batch['lengths'][:n_examples],
        'labels': first_batch['labels'][:n_examples],
        # 'times': first_batch['time'],
        'feature_names': get_feature_names(dataset)
    }
    return out


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('wandb_run', type=str)
    parser.add_argument('--n_samples', type=int, default=200, help="Number of samples to use for the shap computation.")
    parser.add_argument('--hours_before_end', type=int, default=0, help="Number of hours prior to the end of stay to look at for feature importance estimation.")
    parser.add_argument('--min_length', type=int, default=5, help="Minimal length of instance in order to be used for background.")
    parser.add_argument('--max_examples', type=int, default=50, help="Number of instances from dataset to use as background.")

    # parser.add_argument(
    #     '--dataset', required=True, type=str, choices=src.torch.datasets.__all__,
    # )
    # parser.add_argument(
    #     '--split',
    #     default='validation',
    #     choices=['train', 'validation', 'test'],
    #     type=str
    # )
    parser.add_argument('--output', type=str, required=True, help='Output path to store pickle with shap values.')
    params = parser.parse_args()

    model, dataset = get_model_and_dataset(params.wandb_run)
    shap_values = run_shap_analysis(
        model, dataset, hours_before_end=params.hours_before_end,
        n_samples=params.n_samples, min_length=params.min_length,
        max_examples=params.max_examples)
    # print(shap_values)
    import pickle
    with open(params.output, 'wb') as f:
        pickle.dump(shap_values, f)

