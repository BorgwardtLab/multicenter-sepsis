"""Training routines for models."""
from argparse import ArgumentParser, Namespace
import json
from collections import defaultdict
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import src.torch.models
import src.datasets
from src.torch.torch_utils import JsonEncoder, TbWithBestValueLogger


def namespace_without_none(namespace):
    new_namespace = Namespace()
    for key, value in vars(namespace).items():
        if value is not None and type(value) != type:
            if hasattr(value, '__len__'):
                if len(value) == 0:
                    continue
            setattr(new_namespace, key, value)
    return new_namespace


def main(hparams, model_cls):
    """Main function train model."""
    # init module

    model = model_cls(**vars(namespace_without_none(hparams)))
    logger = TbWithBestValueLogger(
        hparams.log_path,
        model.metrics_initial,
        add_best=True,
        name=hparams.exp_name,
        default_hp_metric=False
    )
    checkpoint_dir = os.path.join(
        logger.log_dir, 'checkpoints')

    monitor_score = hparams.monitor
    monitor_mode = hparams.monitor_mode

    model_checkpoint_cb = ModelCheckpoint(
        monitor=monitor_score,
        mode=monitor_mode,
        save_top_k=1,
        dirpath=checkpoint_dir
    )
    early_stopping_cb = EarlyStopping(
        monitor=monitor_score, patience=10, mode=monitor_mode, strict=True,
        verbose=1)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer(
        callbacks=[early_stopping_cb],
        checkpoint_callback=model_checkpoint_cb,
        max_epochs=hparams.max_epochs,
        logger=logger,
        gpus=hparams.gpus
    )
    trainer.fit(model)
    trainer.logger.save()
    print('Loading model with', monitor_mode, monitor_score)
    print(model_checkpoint_cb.best_model_path)
    loaded_model = model_cls.load_from_checkpoint(
        checkpoint_path=model_checkpoint_cb.best_model_path)
    trainer.test(loaded_model)
    trainer.logger.save()
    all_metrics = {**trainer.logger.last, **trainer.logger.best}
    all_metrics = {n: v for n, v in all_metrics.items() if '/' in n}
    prefixes = {name.split('/')[0] for name in all_metrics.keys()}
    results = defaultdict(dict)
    for name, value in all_metrics.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                results[prefix][name.split('/')[1]] = value

    from src.torch.eval_model import online_eval
    masked_result = online_eval(
        loaded_model,
        getattr(src.datasets, hparams.dataset, 'validation'),
        'validation'
    )
    results['validation_masked'] = masked_result
    with open(os.path.join(logger.log_dir, 'result.json'), 'w') as f:
        json.dump(results, f, cls=JsonEncoder)

    print('MASKED TEST RESULTS')
    print({
        key: value for key, value in results['validation_masked'].items()
        if key not in ['labels', 'predictions']
    })

    # Filter out parts of hparams which belong to Hyperargparse
    config = {
        key: value
        for key, value in vars(hparams).items()
        if not callable(value)
    }
    with open(os.path.join(logger.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, cls=JsonEncoder)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--log-path', default='logs')
    parser.add_argument('--exp-name', default='train_torch_model')
    parser.add_argument('--model', choices=src.torch.models.__all__, type=str,
                        default='AttentionModel')
    parser.add_argument('--max-epochs', default=100, type=int)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--hyperparam-draws', default=0, type=int)
    parser.add_argument('--monitor', type=str,
                        default='online_val/loss')
    parser.add_argument('--monitor-mode', type=str, choices=['max', 'min'],
                        default='min')
    parser.add_argument(
        '--feature-set', default='all',
        help='which feature set should be used: [all, challenge], where challenge refers to the subset as derived from physionet challenge variables'
    )

    # figure out which model to use
    temp_args = parser.parse_known_args()[0]

    if temp_args.monitor.endswith('loss') and temp_args.monitor_mode == 'max':
        print(
            'It looks like you are trying to run early stopping on a loss '
            'using the wrong monitor mode (max).')
        print('Exiting...')
        import sys
        sys.exit(1)

    # let the model add what it wants
    model_cls = getattr(src.torch.models, temp_args.model)

    parser = model_cls.add_model_specific_args(parser)
    hparams = parser.parse_args()
    if hparams.hyperparam_draws > 0:
        for hyperparam_draw in hparams.trials(hparams.hyperparam_draws):
            print(hyperparam_draw)
            hyperparam_draw = Namespace(**hyperparam_draw.__getstate__())
            main(hyperparam_draw, model_cls)
    else:
        # Need to do this in order to allow pickling
        hparams = Namespace(**hparams.__getstate__())
        main(hparams, model_cls)
