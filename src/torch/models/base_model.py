"""Base model for all models implementing datasets and training."""
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score)
from test_tube import HyperOptArgumentParser

import src.datasets
from src.evaluation import physionet2019_utility
from src.torch.torch_utils import (
    variable_length_collate, ComposeTransformations, LabelPropagation)


class BaseModel(pl.LightningModule):
    """Base model for all models implementing datasets and training."""

    @property
    def transforms(self):
        return [
            LabelPropagation(-self.hparams.label_propagation)
        ]

    def _get_input_dim(self):
        data = self.dataset_cls(
            split='train',
            transform=ComposeTransformations(self.transforms),
            feature_set=self.hparams.feature_set
        )
        return int(data[0]['ts'].shape[-1])

    @property
    def metrics_initial(self):
        return {
            'train/loss': float('inf'),
            'online_val/loss': float('inf'),
            'online_val/physionet2019_score': float('-inf'),
            'online_val/average_precision': float('-inf'),
            'online_val/auroc': float('-inf'),
            'online_val/balanced_accuracy': float('-inf'),
            'validation/loss': float('inf'),
            'validation/physionet2019_score': float('-inf'),
            'validation/average_precision': float('-inf'),
            'validation/auroc': float('-inf'),
            'validation/balanced_accuracy': float('-inf')
        }

    def __init__(self, dataset, pos_weight, label_propagation, learning_rate,
                 batch_size, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_cls = getattr(src.datasets, self.hparams.dataset)
        d = self.dataset_cls(split='train')
        self.train_indices, self.val_indices = d.get_stratified_split(87346583)
        self.loss = torch.nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=torch.Tensor(
                [self.hparams.pos_weight ])
        )

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
        # per_instance_loss = loss.sum(-1) / n_tp.float()
        n_tp = n_tp.sum()
        per_tp_loss = loss.sum() / n_tp.float()
        self.log('loss', per_tp_loss)
        return {
            'loss': per_tp_loss,
            'n_tp': n_tp,
        }

    def training_epoch_end(self, outputs):
        total_tp = 0
        total_loss = 0
        for x in outputs:
            n_tp = x['n_tp']
            total_tp += n_tp
            total_loss += n_tp * x['loss']

        average_loss = total_loss / total_tp
        self.log('train/loss', average_loss, prog_bar=True)

    def _shared_eval(self, batch, batch_idx, prefix):
        data, lengths, labels = batch['ts'], batch['lengths'], batch['labels']
        output = self.forward(data, lengths).squeeze(-1)

        # Flatten outputs to support nll_loss
        invalid_indices = torch.isnan(labels)
        cloned_labels = labels.clone()
        cloned_labels[invalid_indices] = 0.
        loss = self.loss(output, cloned_labels.float())
        loss[invalid_indices] = 0.
        # Aggregate first over time then over instances
        n_tp = labels.shape[-1] - invalid_indices.sum(-1, keepdim=True)
        # per_instance_loss = loss.sum(-1) / n_tp.float()
        n_tp = n_tp.sum()
        per_tp_loss = loss.sum() / n_tp.float()

        scores = torch.sigmoid(output)
        return {
            f'{prefix}_loss': per_tp_loss.detach(),
            f'{prefix}_n_tp': n_tp,
            f'{prefix}_labels': labels.cpu().detach().numpy(),
            f'{prefix}_scores': scores.cpu().detach().numpy()
        }

    def _shared_end(self, outputs, prefix):
        total_tp = 0
        total_loss = 0
        n_tp_key = f'{prefix}_n_tp'
        loss_key = f'{prefix}_loss'
        for x in outputs:
            n_tp = x[n_tp_key]
            total_tp += n_tp
            total_loss += n_tp * x[loss_key]
        average_loss = total_loss / total_tp

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
        self.log(f'{prefix}/physionet2019_score', physionet_score)
        self.log(f'{prefix}/average_precision', average_precision)
        self.log(f'{prefix}/auroc', auroc)
        self.log(f'{prefix}/balanced_accuracy', balanced_accuracy)
        self.log(f'{prefix}/loss', average_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, 'online_val')

    def validation_epoch_end(self, outputs):
        self._shared_end(outputs, 'online_val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix='validation')

    def test_epoch_end(self, outputs):
        self._shared_end(outputs, 'validation')

    def configure_optimizers(self):
        """Get optimizers."""
        # TODO: We should also add a scheduler here to implement warmup. Most
        # recent version of pytorch lightning seems to have problems with how
        # it was implemented before.
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
        """Get train data loader."""
        return DataLoader(
            Subset(
                self.dataset_cls(
                    split='train',
                    transform=ComposeTransformations(self.transforms),
                    feature_set=self.hparams.feature_set
                ),
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
                    transform=ComposeTransformations(self.transforms),
                    feature_set=self.hparams.feature_set
                ),
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
                transform=ComposeTransformations(self.transforms),
                feature_set=self.hparams.feature_set
            ),
            shuffle=False,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size,
            num_workers=4
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
        parser.add_argument('--pos-weight', type=float, default=1.)
        return parser
