from argparse import ArgumentParser
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score)
from test_tube import HyperOptArgumentParser

import src.datasets
from src.evaluation import physionet2019_utility
from src.torch.models.fixed_lightning import FixedLightningModule
from src.torch.torch_utils import (
    variable_length_collate, ComposeTransformations, LabelPropagation)


class BaseModel(FixedLightningModule):
    def _get_input_dim(self):
        data = self.dataset_cls(
            split='train',
            transform=ComposeTransformations(self.transforms)
        )
        return data[0]['ts'].shape[-1]

    def __init__(self, hparams):
        super().__init__()
        self.dataset_cls = getattr(src.datasets, hparams.dataset)
        d = self.dataset_cls(split='train')
        self.train_indices, self.val_indices = d.get_stratified_split(87346583)
        self.hparams = hparams
        self.loss = torch.nn.BCEWithLogitsLoss(
            reduction='none', pos_weight=torch.Tensor([hparams.pos_weight]))

    def training_step(self, batch, batch_idx):
        """Run a single training step."""
        data, lengths, labels = batch['ts'], batch['lengths'], batch['labels']
        output = self.forward(data, lengths).squeeze(-1)
        invalid_indices = torch.isnan(labels)
        labels[invalid_indices] = 0.
        loss = self.loss(output, labels)
        loss[invalid_indices] = 0.
        # Aggregate first over time then over instances
        n_tp = labels.shape[-1] - invalid_indices.sum(-1, keepdim=True)
        per_instance_loss = loss.sum(-1) / n_tp.float()
        return {'loss': per_instance_loss.mean(), 'n_samples': data.shape[0]}

    def training_epoch_end(self, outputs):
        total_samples = 0
        total_loss = 0
        for x in outputs:
            n_samples = x['n_samples']
            total_samples += n_samples
            total_loss += n_samples * x['loss']

        average_loss = total_loss / total_samples
        return {
            'log': {'train_loss': average_loss},
            'progress_bar': {'train_loss': average_loss}
        }

    def _shared_eval(self, batch, batch_idx, prefix):
        data, lengths, labels = batch['ts'], batch['lengths'], batch['labels']
        output = self.forward(data, lengths).squeeze(-1)

        # Flatten outputs to support nll_loss
        n_val = labels.shape[0]
        invalid_indices = torch.isnan(labels)
        cloned_labels = labels.clone()
        cloned_labels[invalid_indices] = 0.
        loss = self.loss(output, cloned_labels.float())
        loss[invalid_indices] = 0.
        # Aggregate first over time then over instances
        n_tp = labels.shape[-1] - invalid_indices.sum(-1, keepdim=True)
        per_instance_loss = loss.sum(-1) / n_tp.float()

        scores = torch.sigmoid(output)
        return {
            f'{prefix}_loss': per_instance_loss.detach().mean(),
            f'{prefix}_n': n_val,
            f'{prefix}_labels': labels.cpu().detach().numpy(),
            f'{prefix}_scores': scores.cpu().detach().numpy()
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
        scores = []
        for x in outputs:
            cur_labels = x[f'{prefix}_labels']
            cur_scores = x[f'{prefix}_scores']
            cur_preds = (cur_scores >= 0.5).astype(float)
            for label, pred, score in zip(cur_labels, cur_preds, cur_scores):
                selection = ~np.isnan(label)
                # Get index of first invalid label, this allows the labels to
                # have gaps with NaN in between.
                first_invalid_label = (
                    len(selection) - np.argmax(selection[::-1]))
                labels.append(label[:first_invalid_label])
                predictions.append(pred[:first_invalid_label])
                scores.append(score[:first_invalid_label])

        physionet_score = physionet2019_utility(
            labels, predictions, shift_labels=self.hparams.label_propagation)
        # Scores below require flattened predictions
        labels = np.concatenate(labels, axis=0)
        is_valid = ~np.isnan(labels)
        labels = labels[is_valid]
        predictions = np.concatenate(predictions, axis=0)[is_valid]
        scores = np.concatenate(scores, axis=0)[is_valid]
        average_precision = average_precision_score(labels, scores)
        auroc = roc_auc_score(labels, scores)
        balanced_accuracy = balanced_accuracy_score(labels, predictions)
        data = {
            f'{prefix}_loss': average_loss.cpu().detach(),
            f'{prefix}_average_precision': torch.as_tensor(average_precision),
            f'{prefix}_auroc': torch.as_tensor(auroc),
            f'{prefix}_balanced_accuracy': torch.as_tensor(balanced_accuracy),
            f'{prefix}_physionet2019_score': torch.as_tensor(physionet_score)
        }
        return {
            'progress_bar': data,
            'log': data
        }

    @property
    def transforms(self):
        return [
            LabelPropagation(-self.hparams.label_propagation)
        ]

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, 'online_val')

    def validation_epoch_end(self, outputs):
        return self._shared_end(outputs, 'online_val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix='val')

    def test_epoch_end(self, outputs):
        return self._shared_end(outputs, 'val')

    def configure_optimizers(self):
        """Get optimizers."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
        """Get train data loader."""
        return DataLoader(
            Subset(
                self.dataset_cls(
                    split='train',
                    transform=ComposeTransformations(self.transforms)),
                self.train_indices
            ),
            shuffle=True,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        """Get validation data loader."""
        return DataLoader(
            Subset(
                self.dataset_cls(
                    split='train',
                    transform=ComposeTransformations(self.transforms)),
                self.val_indices
            ),
            shuffle=False,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        """Get test data loader."""
        return DataLoader(
            self.dataset_cls(
                split='validation',
                transform=ComposeTransformations(self.transforms)),
            shuffle=False,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        """Specify the hyperparams."""
        parser = HyperOptArgumentParser(
            strategy='random_search', parents=[parent_parser])
        # training specific
        parser.add_argument(
            '--dataset', type=str, choices=src.datasets.__all__,
            default='PreprocessedDemoDataset'
        )
        parser.opt_range(
            '--learning-rate', default=0.01, type=float,
            tunable=True, log_base=10., low=0.0001, high=0.01
        )
        parser.opt_list(
            '--batch-size', default=32, type=int,
            options=[16, 32, 64, 128, 256],
            tunable=True
        )
        parser.add_argument('--label-propagation', default=6, type=int)
        parser.add_argument('--pos-weight', type=float, default=50.)
        return parser
