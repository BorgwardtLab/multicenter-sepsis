"""Script to load trained model and run shape analysis."""

import tempfile
import os
import torch
import sys
import wandb
import warnings

import numpy as np

from src.torch.eval_model import device

from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler

from src.torch.shap_utils import get_model_and_dataset

import shap

wandb_api = wandb.Api()


# Shap evaluation perturbs the inputs, thus we cannot rely on the length
# argument.  Instead use Nan padding and reconstruct length in model.
def variable_length_collate_nan_padding(batch, lengths=None):
    """Combine multiple instances of irregular lengths into a batch.

    Parameters
    ----------
    batch
        Input batch

    lengths : `torch.tensor`, optional
        If provided, will take the place of the pre-defined lengths in
        the batch. This can be used to perform post-hoc masking, which
        we require after finding the maximum prediction score index of
        each sample.
    """
    # This converts the batch from [{'a': 1, 'b': 2}, ..] format to
    # {'a': [1], 'b': [2]}
    transposed_items = {
        key: list(map(lambda a: a[key], batch))
        for key in batch[0].keys()
    }

    transposed_items['id'] = np.array(transposed_items['id'])

    if lengths is None:
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

    def __init__(self, model):
        """Wrap a model to only return output at the end of the stay.

        Args:
            model: Model to wrap
        """
        super().__init__()
        self.model = model

    def forward(self, x, lengths=None, statics=None, return_indices=False):
        """Perform forward pass through model."""
        mask = None
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

            # Remove the nan values again prior to model input. We will
            # re-use the mask later on (if it exists) to remove all
            # predictions that go beyond the length.
            mask = torch.isnan(x)
            x = torch.where(mask, torch.zeros_like(x), x)

        assert mask is not None

        out = self.model(x, lengths=lengths, statics=statics)

        # Remove all predictions *after* the length of the time series,
        # thus ensuring that we do not ask for explanations at the wrong
        # places.
        out = torch.where(
            mask.any(2).unsqueeze(-1),
            torch.zeros_like(out),
            out
        )

        # Pick *largest* prediction score along each time series.
        index = out.argmax(1).squeeze()
        assert torch.all(index >= 0)
        out = out[torch.arange(out.shape[0]), index]

        if return_indices:
            return out, index
        else:
            return out


def run_shap_analysis(
    model,
    dataset,
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
    print('Lengths:', data['lengths'])

    def get_model_inputs(batch):
        return [batch['ts'].to(device)]

    sample_dataset = get_model_inputs(data)
    wrapped_model = ModelWrapper(model)
    explainer = shap.GradientExplainer(
        wrapped_model,
        sample_dataset,
    )

    # Get the index at which the maximum prediction happens. This will
    # enable us to mask the time series afterwards.
    _, indices = wrapped_model(sample_dataset[0], return_indices=True)
    indices = indices.cpu().numpy()

    # Re-pad the time series with the detected lengths to ensure that we
    # are not trying to explain *future* time steps with old information
    # from past time points.
    for sample, index in zip(sample_dataset[0], indices):
        sample[index + 1:, :] = np.nan

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

    parser.add_argument(
        '--min_length',
        type=int,
        default=0,
        help='Minimal length of instance in order to be used for background.'
    )

    params = parser.parse_args()

    if os.path.exists(params.output) and not params.force:
        warnings.warn(
            f'Refusing to overwrite "{params.output}" without `--force`.'
        )
        sys.exit(-1)

    model, dataset = get_model_and_dataset(params.wandb_run)

    shap_values = run_shap_analysis(
        model, dataset,
        n_samples_data=params.n_samples,
        n_samples_background=params.n_samples_background,
        min_length=params.min_length,
    )

    import pickle
    with open(params.output, 'wb') as f:
        pickle.dump(shap_values, f)
