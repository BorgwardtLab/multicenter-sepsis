import json
import tempfile
import wandb
from src.torch.eval_model import online_eval, compute_md5hash, device
import src.torch.datasets
import src.torch.models

wandb_api = wandb.Api()

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
    return run, {
        "model": run_info['model'],
        "run_id": run_path,
        "model_path": checkpoint_path,
        "model_checksum": model_checksum,
        "model_params": run_info,
        "dataset_train": run_info['dataset'],
        "task": run_info['task'],
        "label_propagation": run_info['label_propagation'],
        "rep": run_info['rep']
    }


def main(run_id, dataset, split, output):
    """Main function to evaluate a model."""
    with tempfile.TemporaryDirectory() as tmp:
        # Download checkpoint to temporary directory
        run, out = extract_model_information(run_id, tmp)
        out['dataset_eval'] = dataset
        out['split'] = split

        model_cls = getattr(src.torch.models, out['model'])
        dataset_cls = getattr(src.torch.datasets, dataset)
        model = model_cls.load_from_checkpoint(
            out['model_path'],
            dataset=dataset
        )
    model.to(device)
    eval_results = online_eval(
        model, dataset_cls, split, check_matching_unmasked=True)
    out.update(eval_results)

    print({
        key: value for key, value in out.items()
        if key not in ['targets', 'labels', 'predictions', 'scores', 'ids', 'times']
    })

    if output is not None:
        with open(output, 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('wandb_run', type=str)
    parser.add_argument(
        '--dataset', required=True, type=str, choices=src.torch.datasets.__all__,
    )
    parser.add_argument(
        '--split',
        default='validation',
        choices=['train', 'validation', 'test'],
        type=str
    )
    parser.add_argument('--output', type=str, default=None)
    params = parser.parse_args()

    main(
        params.wandb_run,
        params.dataset,
        params.split,
        params.output,
    )


