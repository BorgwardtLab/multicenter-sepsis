"""Training routines for models."""
from argparse import ArgumentParser, Namespace
import os
import pytorch_lightning as pl

import src.torch.models


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

    model_checkpoint_cb = pl.callbacks.model_checkpoint.ModelCheckpoint(
        os.path.join(checkpoint_dir, '{epoch}-{online_val_physionet2019_score:.2f}'),
        monitor='online_val_physionet2019_score',
        mode='max'
    )
    early_stopping_cb = pl.callbacks.early_stopping.EarlyStopping(
        monitor='online_val_physionet2019_score', patience=10, mode='max', strict=True)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer(
        checkpoint_callback=model_checkpoint_cb,
        early_stop_callback=early_stopping_cb,
        max_epochs=hparams.max_epochs,
        logger=logger,
        gpus=hparams.gpus
    )
    trainer.fit(model)
    trainer.logger.save()
    print('Loading model with best physionet score...')
    checkpoints = os.listdir(checkpoint_dir)
    assert len(checkpoints) == 1
    last_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    loaded_model = model_cls.load_from_checkpoint(last_checkpoint)
    trainer.test(loaded_model)
    trainer.logger.save()


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--log-path', default='logs')
    parser.add_argument('--exp-name', default='train_torch_model')
    parser.add_argument('--model', choices=src.torch.models.__all__, type=str,
                        default='AttentionModel')
    parser.add_argument('--max-epochs', default=100, type=int)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--hyperparam-draws', default=0, type=int)
    # figure out which model to use
    temp_args = parser.parse_known_args()[0]

    # let the model add what it wants
    model_cls = getattr(src.torch.models, temp_args.model)

    parser = model_cls.add_model_specific_args(parser)
    hparams = parser.parse_args()
    if hparams.hyperparam_draws > 0:
        for hyperparam_draw in hparams.trials(hparams.hyperparam_draws):
            print(hyperparam_draw)
            main(hyperparam_draw, model_cls)
    else:
        main(hparams, model_cls)
