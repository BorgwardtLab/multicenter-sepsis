"""Training routines for models."""
from argparse import ArgumentParser, Namespace
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.data import ComposeTransformations, PositionalEncoding, \
    to_observation_tuples
from src.torch.torch_utils import variable_length_collate
from src.torch.models import AttentionModel
from src.evaluation import physionet2019_utility
import src.datasets


class PlAttentionModel(pl.LightningModule):
    transform = ComposeTransformations([
        PositionalEncoding(1, 250000, 20),  # apply positional encoding
        to_observation_tuples               # mask nan with zero add indicators
    ])

    def _get_input_dim(self):
        data = self.dataset_cls(
            split_repetition=self.hparams.split_repetition,
            split='train',
            transform=self.transform
        )
        return data[0]['ts'].shape[-1]


    def __init__(self, hparams):
        super().__init__()
        self.dataset_cls = getattr(src.datasets, hparams.dataset)
        self.hparams = hparams
        ts_dim = self.val_dataloader()
        self.model = AttentionModel(
            d_in=self._get_input_dim(),
            d_model=hparams.d_model,
            n_layers=hparams.n_layers,
            n_heads=hparams.n_heads,
            qkv_dim=hparams.qkv_dim
        )

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        """Run a single training step."""
        data, labels = batch['ts'], batch['labels']
        output = self.forward(data)
        # Flatten outputs to support nll_loss
        output = output.reshape(-1, 2)
        labels = labels.reshape(-1)

        # approximate computation of class imbalance using batch
        label_weight = (
            labels[:, None] ==
            torch.tensor([0, 1], device=labels.device)[None, :]
        )

        label_weight = label_weight.sum(dim=0, dtype=torch.float)
        label_weight = label_weight.sum() / label_weight

        loss = F.nll_loss(output, labels, weight=label_weight)
        return {'loss': loss}

    def _shared_eval(self, batch, batch_idx, prefix):
        data, labels = batch['ts'], batch['labels']
        y_hat = self.forward(data)

        # Flatten outputs to support nll_loss
        y_hat_flat = y_hat.reshape(-1, 2)
        labels_flat = labels.reshape(-1)
        n_val = labels.shape[0]

        labels = labels.numpy()
        y_hat = y_hat.numpy()
        labels_list = []
        y_hat_list = []
        for label, pred in zip(labels, y_hat):
            selection = label != -100
            labels_list.append(label[selection])
            y_hat_list.append(pred[selection])

        return {
            f'{prefix}_loss': F.nll_loss(y_hat_flat, labels_flat),
            f'{prefix}_n': n_val,
            f'{prefix}_labels': labels_list,
            f'{prefix}_predictions': y_hat_list
        }

    def _shared_end(self, outputs, prefix):
        total_samples = sum(x[f'{prefix}_n'] for x in outputs)
        val_loss_mean = (
            torch.stack([
                x[f'{prefix}_loss'] * x[f'{prefix}_n'] for x in outputs
            ]).sum() / total_samples
        )
        labels = []
        predictions = []
        for x in outputs:
            labels.extend(x[f'{prefix}_labels'])
            predictions.extend(
                [np.argmax(el, axis=-1) for el in x[f'{prefix}_predictions']])

        physionet_score = physionet2019_utility(labels, predictions)
        return {
            f'{prefix}_loss': val_loss_mean,
            f'{prefix}_physionet2019_score': physionet_score
        }

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, 'val')

    def validation_epoch_end(self, outputs):
        return self._shared_end(outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix='val')

    def test_epoch_end(self, outputs):
        return self._shared_end(outputs, 'tests')

    def configure_optimizers(self):
        """Get optimizers."""
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        """Get train data loader."""
        return DataLoader(
            self.dataset_cls(
                split_repetition=self.hparams.split_repetition,
                split='train',
                transform=self.transform),
            shuffle=True,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size
        )

    def val_dataloader(self):
        """Get validation data loader."""
        return DataLoader(
            self.dataset_cls(
                split_repetition=self.hparams.split_repetition,
                split='validation',
                transform=self.transform),
            shuffle=False,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size
        )

    def test_dataloader(self):
        """Get test data loader."""
        return DataLoader(
            self.dataset_cls(
                split_repetition=self.hparams.split_repetition,
                split='test',
                transform=self.transform),
            shuffle=False,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Specify the hyperparams."""
        parser = ArgumentParser(parents=[parent_parser])
        # training specific
        parser.add_argument(
            '--dataset', type=str, choices=src.datasets.__all__,
            default='Physionet2019Dataset'
        )
        parser.add_argument(
            '--split-repetition', type=int, choices=[0, 1, 2, 3, 4], default=0)
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)

        # MODEL specific
        parser.add_argument('--d-model', type=int, default=64)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--n-heads', type=int, default=8)
        parser.add_argument('--qkv-dim', type=int, default=32)
        return parser


def namespace_without_none(namespace):
    new_namespace = Namespace()
    for key, value in vars(namespace).items():
        if value is not None and value != [] and type(value) != type:
            setattr(new_namespace, key, value)
    return new_namespace


def main(hparams):
    """Main function train model."""
    # init module
    model = PlAttentionModel(namespace_without_none(hparams))

    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(model)
    trainer.logger.save()


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)
    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = PlAttentionModel.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()
    main(hparams)
