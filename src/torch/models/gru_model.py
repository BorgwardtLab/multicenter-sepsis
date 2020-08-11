"""Attention based model for sepsis."""
import torch
import torch.nn as nn
from src.torch.models.base_model import BaseModel
from src.torch.torch_utils import to_observation_tuples


class GRUModel(BaseModel):
    """GRU Model."""

    def __init__(self, hparams):
        """GRUModel.

        Args:
            d_model: Dimensionality of hidden state
            n_layers: Number of stacked GRU layers
            dropout: Fraction of elements that should be dropped out
        """
        super().__init__(hparams)
        d_model = hparams.d_model
        n_layers = hparams.n_layers
        dropout = hparams.dropout
        d_in = self._get_input_dim()

        self.gru = nn.GRU(
            hidden_size=d_model,
            input_size=d_in,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True)
        self.linear = nn.Linear(d_model, 1)

    @property
    def transforms(self):
        parent_transforms = super().transforms
        parent_transforms.append(
            to_observation_tuples  # mask nan with zero and add indicator
        )
        return parent_transforms

    def forward(self, x, lengths):
        """Apply GRU model to input x.
            - x: is a three dimensional tensor (batch, stream, channel)
            - lengths: is a one dimensional tensor (batch,) giving the true
              length of each batch element along the stream dimension
        """
        # Convert padded sequence to packed sequence for increased efficiency
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        x, hidden_states = self.gru(x)
        x = torch.nn.utils.rnn.PackedSequence(
            self.linear(x.data),
            x.batch_sizes,
            x.sorted_indices,
            x.unsorted_indices
        )
        # Unpack sequence to padded tensor for subsequent operations
        x, lenghts = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        return x

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        # MODEL specific
        parser.opt_list(
            '--d-model', type=int, default=64,
            tunable=True, options=[64, 128, 256, 512]
        )
        parser.opt_list(
            '--n-layers', type=int, default=1,
            tunable=True, options=[1, 2, 3]
        )
        parser.opt_list(
            '--dropout', default=0.1, type=float,
            tunable=True, options=[0., 0.1, 0.2, 0.3, 0.4]
        )
        return parser
