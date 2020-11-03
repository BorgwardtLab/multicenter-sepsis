"""Training routines for models."""
from argparse import ArgumentParser, Namespace
import json
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import src.torch.models
import src.datasets
from src.torch.torch_utils import JsonEncoder


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

    model = model_cls(namespace_without_none(hparams))
    logger = pl.loggers.TestTubeLogger(
        hparams.log_path, name=hparams.exp_name)
    exp = logger.experiment
    exp_path = exp.get_data_path(exp.name, exp.version)
    checkpoint_dir = os.path.join(
        exp_path, 'checkpoints')

    monitor_score = hparams.monitor
    monitor_mode = hparams.monitor_mode

    model_checkpoint_cb = ModelCheckpoint(
        os.path.join(checkpoint_dir, '{epoch}-{'+monitor_score+':.2f}'),
        monitor=monitor_score,
        mode=monitor_mode
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
    checkpoints = os.listdir(checkpoint_dir)
    assert len(checkpoints) == 1
    last_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    loaded_model = model_cls.load_from_checkpoint(
        checkpoint_path=last_checkpoint)
    trainer.test(loaded_model)
    trainer.logger.save()
    last_metrics = trainer.logger.experiment.metrics[-1]
    from src.torch.eval_model import online_eval
    masked_result = online_eval(
        loaded_model,
        getattr(src.datasets, hparams.dataset, 'validation'),
        'validation'
    )
    result = {}
    result['validation'] = {
        key.replace('val_', ''): value for key, value in last_metrics.items()}
    result['validation_masked'] = masked_result
    with open(os.path.join(exp_path, 'result.json'), 'w') as f:
        json.dump(result, f, cls=JsonEncoder)

    print('MASKED TEST RESULTS')
    print({
        key: value for key, value in result['validation_masked'].items()
        if key not in ['labels', 'predictions']
    })

    # Filter out parts of hparams which belong to Hyperargparse
    config = {
        key: value
        for key, value in vars(hparams).items()
        if not callable(value)
    }
    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
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
                        default='online_val_loss')
    parser.add_argument('--monitor-mode', type=str, choices=['max', 'min'],
                        default='min')
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
