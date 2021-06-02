"""Script to load trained model and run shape analysis."""
import json
import tempfile
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.torch.eval_model import compute_md5hash, device
from src.torch.torch_utils import ComposeTransformations, variable_length_collate
import src.torch.datasets
import src.torch.models
import shap

wandb_api = wandb.Api()

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
    transposed_items['labels_shifted'] = transposed_items['labels_shifted'].astype(np.float32)
    transposed_items['targets'] = transposed_items['targets'].astype(np.float32)

    return {
        key: torch.from_numpy(data) for key, data in
        transposed_items.items()
    }


def extract_model_information(run_path, tmp):
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


def get_model_and_dataset(run_id, output):
    """Main function to evaluate a model."""
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


def run_shap_analysis(model, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=50,
        shuffle=False,
        collate_fn=variable_length_collate_nan_padding,
    )

    def get_model_inputs(batch):
        return [batch['ts'].to(device)]

    first_batch = dataloader.__iter__().__next__()
    sample_dataset = get_model_inputs(first_batch)
    explainer = shap.GradientExplainer(model, sample_dataset, batch_size=50)
    shap_values = explainer.shap_values(
        [sample_dataset[0][:2]])
    return shap_values




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('wandb_run', type=str)
    # parser.add_argument(
    #     '--dataset', required=True, type=str, choices=src.torch.datasets.__all__,
    # )
    # parser.add_argument(
    #     '--split',
    #     default='validation',
    #     choices=['train', 'validation', 'test'],
    #     type=str
    # )
    parser.add_argument('--output', type=str, required=True)
    params = parser.parse_args()

    model, dataset = get_model_and_dataset(
        params.wandb_run,
        params.output,
    )
    shap_values = run_shap_analysis(model, dataset)
    print(shap_values)
    import pickle
    with open(params.output, 'wb') as f:
        pickle.dump(shap_values, f)



