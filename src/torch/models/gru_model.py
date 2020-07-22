"""Attention based model for sepsis."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.torch.models.base_model import BaseModel


class GRUModel(BaseModel):
    """GRU Model"""

    def __init__(self, hparams):
        """GRUModel.

        Args:
            
        """
        super().__init__(hparams)
        d_model = hparams.d_model
        n_layers = hparams.n_layers
        dropout = hparams.dropout
        d_in = self._get_input_dim()
        
        kwargs = {'hidden_size': d_model, 'input_size': d_in}
        kwargs['batch_first'] = True
        kwargs['bidirectional'] = False
        kwargs['num_layers'] = n_layers

        self.gru = nn.GRU(**kwargs)
        self.linear = nn.Linear(d_model, 1)      
   
    @property
    def transforms(self):
        parent_transforms = super().transforms
        return parent_transforms

    # came until here.. 
    def forward(self, x, lengths):
        """Apply GRU model to input x.
            - x: is a three dimensional tensor (batch, stream, channel)
            - lengths: is a one dimensional tensor (batch,) giving the true length of each batch element along the
                stream dimension
        """
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        #for starters we work with last hidden state, could also use last output or concat
        _, x = self.gru(x)
        x = x[-1]  # take the last GRU layer
        x = self.linear(x)
        return x
  
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        # MODEL specific
        parser.add_argument('--d-model', type=int, default=64)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--dropout', default=0.1, type=float)
        return parser
