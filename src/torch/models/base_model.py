"""Base model for all models implementing datasets and training."""
import argparse
import numpy as np
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score)
# from test_tube import HyperOptArgumentParser

import src.torch.datasets
from src.torch.datasets import ComposedDataset
from src.evaluation import physionet2019_utility
from src.torch.torch_utils import (
    variable_length_collate, ComposeTransformations, LabelPropagation)
from src.torch.cli_utils import str2bool


class BaseModel(pl.LightningModule):
    """Base model for all models implementing datasets and training."""

    @property
    def transforms(self):
        if self.hparams.task == 'classification':
            return [
                LabelPropagation(shift_left=-self.hparams.label_propagation,
                                 shift_right=self.hparams.label_propagation_right, 
                                 keys=['targets', 'labels_shifted'])
            ]
        else:
            return [
                LabelPropagation(shift_left=-self.hparams.label_propagation,
                                 shift_right=self.hparams.label_propagation_right, 
                                 keys=['labels_shifted'])
            ]

    def _get_input_dim(self):
        data = self.dataset_cls(
            split='train',
            transform=ComposeTransformations(self.transforms),
            **self.dataset_kwargs
        )
        return int(data[0]['ts'].shape[-1])

    def _get_input_dims(self):
        """Get dim of statics and time series."""
        data = self.dataset_cls(
            split='train',
            transform=ComposeTransformations(self.transforms),
            **self.dataset_kwargs
        )[0]
        return int(data['statics'].shape[0]), int(data['ts'].shape[-1])

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
                 batch_size, weight_decay, task='classification', 
                 dataset_kwargs={}, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if isinstance(self.hparams.dataset, (list, tuple)):
            print('Using composed dataset:', self.hparams.dataset)
            # Handle composed datasets, i.e. training on multiple datasets at
            # the same time.
            self.dataset_cls = partial(
                ComposedDataset,
                datasets=[getattr(src.torch.datasets, d) for d in self.hparams.dataset]
            )
        else:
            self.dataset_cls = getattr(src.torch.datasets, self.hparams.dataset)
        d = self.dataset_cls(split='train', **dataset_kwargs)
        self.train_indices, self.val_indices = d.get_stratified_split(87346583)
        self.task = task
        self.dataset_kwargs = dataset_kwargs
        self.lam = d.lam
        if task == 'classification':
            self.loss = torch.nn.BCEWithLogitsLoss(
                reduction='none',
                pos_weight=torch.Tensor(
                    [self.hparams.pos_weight * d.class_imbalance_factor])
            )
        elif task == 'regression':
            self.loss = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError(f'{task} is not a valid task among: [classification, regression]')

    def training_step(self, batch, batch_idx):
        """Run a single training step."""
        statics, data, lengths, targets = batch['statics'], batch['ts'], batch['lengths'], batch['targets']
        statics = None if self.hparams.ignore_statics else statics
        output = self.forward(data, lengths, statics=statics).squeeze(-1)
        invalid_indices = torch.isnan(targets)
        targets[invalid_indices] = 0.
        loss = self.loss(output, targets)
        loss[invalid_indices] = 0.
        # Aggregate first over time then over instances
        n_tp = targets.shape[-1] - invalid_indices.sum(-1, keepdim=True)
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
        statics, data, lengths, targets, labels, labels_shifted = (
            batch['statics'],
            batch['ts'],
            batch['lengths'],
            batch['targets'],
            batch['labels'],
            batch['labels_shifted'])
        statics = None if self.hparams.ignore_statics else statics
        output = self.forward(data, lengths, statics=statics).squeeze(-1)

        # Flatten outputs to support nll_loss
        invalid_indices = torch.isnan(targets)
        cloned_targets = targets.clone()
        cloned_targets[invalid_indices] = 0.
        loss = self.loss(output, cloned_targets.float())
        loss[invalid_indices] = 0.
        # Aggregate first over time then over instances
        n_tp = targets.shape[-1] - invalid_indices.sum(-1, keepdim=True)
        # per_instance_loss = loss.sum(-1) / n_tp.float()
        n_tp = n_tp.sum()
        per_tp_loss = loss.sum() / n_tp.float()

        return {
            f'{prefix}_loss': per_tp_loss.detach(),
            f'{prefix}_n_tp': n_tp,
            f'{prefix}_labels': labels.cpu().detach().numpy(),
            f'{prefix}_labels_shifted': labels_shifted.cpu().detach().numpy(),
            f'{prefix}_scores': output.cpu().detach().numpy()
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
        labels_shifted = []
        predictions = []
        scores = []
        for x in outputs:
            cur_labels = x[f'{prefix}_labels']
            cur_labels_shifted = x[f'{prefix}_labels_shifted']
            cur_scores = x[f'{prefix}_scores']
            # TODO: Why should these be float?
            cur_preds = (cur_scores >= 0).astype(float)
            for label, label_shifted, pred, score in zip(cur_labels, 
                    cur_labels_shifted, cur_preds, cur_scores):
                selection = ~np.isnan(label)
                # Get index of first invalid label, this allows the labels to
                # have gaps with NaN in between.
                first_invalid_label = (
                    len(selection) - np.argmax(selection[::-1]))
                labels.append(label[:first_invalid_label])
                labels_shifted.append(label_shifted[:first_invalid_label])
                predictions.append(pred[:first_invalid_label])
                scores.append(score[:first_invalid_label])

        #shift_labels = (
        #    self.hparams.label_propagation if self.hparams.task == 'classification'
        #    else 0
        #)
        # Since labels are not shifted, no need for accounting for shift with shift_labels
        physionet_score = physionet2019_utility(
            labels, predictions,
            shift_labels=0, lam=self.lam)

        # Scores below require flattened predictions
        # we use shifted labels here:
        labels_shifted = np.concatenate(labels_shifted, axis=0)
        is_valid = ~np.isnan(labels_shifted)
        labels_shifted = labels_shifted[is_valid]
        predictions = np.concatenate(predictions, axis=0)[is_valid]
        scores = np.concatenate(scores, axis=0)[is_valid]
        average_precision = average_precision_score(labels_shifted, scores)
        auroc = roc_auc_score(labels_shifted, scores)
        balanced_accuracy = balanced_accuracy_score(labels_shifted, predictions)
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

        # we don't apply weight decay to bias and layernorm parameters, as inspired by:
        # https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/04-transformers-text-classification.ipynb
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ] 
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def train_dataloader(self):
        """Get train data loader."""
        return DataLoader(
            Subset(
                self.dataset_cls(
                    split='train',
                    transform=ComposeTransformations(self.transforms),
                    **self.dataset_kwargs
                ),
                self.train_indices
            ),
            shuffle=True,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            pin_memory=False
        )

    def val_dataloader(self):
        """Get validation data loader."""
        return DataLoader(
            Subset(
                self.dataset_cls(
                    split='train',
                    transform=ComposeTransformations(self.transforms),
                    **self.dataset_kwargs
                ),
                self.val_indices
            ),
            shuffle=False,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            pin_memory=False
        )

    def test_dataloader(self):
        """Get test data loader."""
        return DataLoader(
            self.dataset_cls(
                split='validation',
                transform=ComposeTransformations(self.transforms),
                **self.dataset_kwargs
            ),
            shuffle=False,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size,
            num_workers=8
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        """Specify the hyperparams."""
        parser = argparse.ArgumentParser(parents=[parent_parser])
        # training specific
        parser.add_argument(
            '--dataset', type=str, choices=src.torch.datasets.__all__,
            nargs='+',
            default='MIMICDemo'
        )
        parser.add_argument(
            '--learning_rate', default=0.001, type=float,
            # tunable=True, log_base=10., low=0.0001, high=0.01
        )
        parser.add_argument(
            '--batch_size', default=32, type=int,
            # options=[16, 32, 64, 128, 256], tunable=True
        )
        parser.add_argument('--weight_decay', default=0., type=float)
        parser.add_argument('--label_propagation', default=6, type=int)
        parser.add_argument('--label_propagation_right', default=24, type=int)
        parser.add_argument('--pos_weight', type=float, default=1.)
        parser.add_argument('--ignore_statics', default=False, type=str2bool)
        return parser
