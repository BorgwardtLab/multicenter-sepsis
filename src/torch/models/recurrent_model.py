"""Recurrent models for sepsis prediction."""
import torch
import torch.nn as nn
from src.torch.models.base_model import BaseModel
from src.torch.torch_utils import add_indicators


class RecurrentModel(BaseModel):
    """Recurrent Model."""

    def __init__(self, model_name, d_model, n_layers, dropout, indicators=False, **kwargs):
        """Recurrent Model.

        Args:
            -model_name: ['GRU', 'LSTM', 'RNN']
            -d_model: Dimensionality of hidden state
            -n_layers: Number of stacked RNN layers
            -dropout: Fraction of elements that should be dropped out,
            -indicators: flag whether missingness indicator transform 
                should be applied on the fly.
        """
        super().__init__(**kwargs)
        self.indicators = indicators
        self.save_hyperparameters()
        model_cls = getattr(torch.nn, model_name)
        d_statics, d_in = self._get_input_dims()
        if not self.hparams.ignore_statics:
            self.initial_state = nn.Linear(d_statics, d_model * n_layers)

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
        if self.indicators:
            parent_transforms.append(
                add_indicators  # mask nan with zero and add indicator
            )
        return parent_transforms

    def forward(self, x, lengths, statics=None):
        """Apply GRU model to input x.
            - x: is a three dimensional tensor (batch, stream, channel)
            - lengths: is a one dimensional tensor (batch,) giving the true
              length of each batch element along the stream dimension
        """
        # Convert padded sequence to packed sequence for increased efficiency
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        hidden_init = None
        if statics is not None:
            hidden_init = self.initial_state(statics)
            hidden_init = hidden_init.view(
                -1, self.hparams.n_layers, self.hparams.d_model)
            hidden_init = hidden_init.permute(1, 0, 2)

        x, hidden_states = self.model(x, hidden_init)
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
        parser.add_argument(
            '--d_model', type=int, default=64,
            # tunable=True, options=[64, 128, 256, 512]
        )
        parser.add_argument(
            '--n_layers', type=int, default=1,
            # tunable=True, options=[1, 2, 3]
        )
        parser.add_argument(
            '--dropout', default=0.1, type=float,
            # tunable=True, options=[0., 0.1, 0.2, 0.3, 0.4]
        )
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

