"""Recurrent models for sepsis prediction."""
import torch
import torch.nn as nn
from src.torch.models.base_model import BaseModel
from src.torch.torch_utils import add_indicators, to_observation_tuples, forward_fill


class RecurrentModel(BaseModel):
    """Recurrent Model."""

    def __init__(self, model_name, d_model, n_layers, dropout, drop_time, **kwargs):
        """Recurrent Model.

        Args:
            - model_name: ['GRU', 'LSTM', 'RNN']
            - d_model: Dimensionality of hidden state
            - n_layers: Number of stacked RNN layers
            - dropout: Fraction of elements that should be dropped out
            - drop_time: Do not feed the time as an additional input
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        model_cls = getattr(torch.nn, model_name)
        d_in = self._get_input_dim()

        self.model = model_cls(
            hidden_size=d_model,
            input_size=d_in,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True)
        self.linear = nn.Linear(d_model, 1)

    @property
    def transforms(self):
        parent_transforms = super().transforms
        if self.hparams.add_time:
            # mask nan with zero and add indicator, also add time as a feature
            parent_transforms.append(
                to_observation_tuples
            )
        elif self.hparams.forward_fill:
            parent_transforms.append(
                forward_fill
            )
        else:
            parent_transforms.append(
                add_indicators
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
        x, hidden_states = self.model(x)
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
        parser.add_argument('--add-time', default=False, action='store_true')
        parser.add_argument('--forward-fill', default=False, action='store_true')
        return parser


class GRUModel(RecurrentModel):
    """
    GRU Model.
    """

    def __init__(self, **kwargs):
        model_name = 'GRU'
        if 'model_name' in kwargs.keys():
            del kwargs['model_name']
        super().__init__(model_name, **kwargs)
        self.save_hyperparameters()


class LSTMModel(RecurrentModel):
    """
    LSTM Model.
    """

    def __init__(self, **kwargs):
        model_name = 'LSTM'
        if 'model_name' in kwargs.keys():
            del kwargs['model_name']
        super().__init__(model_name, **kwargs)
        self.save_hyperparameters()


class RNNModel(RecurrentModel):
    """
    RNN Model.
    """

    def __init__(self, **kwargs):
        model_name = 'RNN'
        if 'model_name' in kwargs.keys():
            del kwargs['model_name']
        super().__init__(model_name, **kwargs)
        self.save_hyperparameters()
