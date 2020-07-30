"""Attention based model for sepsis."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.torch.models.base_model import BaseModel
from src.torch.torch_utils import PositionalEncoding, to_observation_tuples


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, n_features = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class BatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        # NB: pytorch bn mom is opposite of what you'd expect
        self.mom, self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (1, 1, nf))
        self.adds  = nn.Parameter(torch.zeros(1, 1, nf))
        self.register_buffer('vars',  torch.ones(1, 1,nf))
        self.register_buffer('means', torch.zeros(1, 1,nf))

    def update_stats(self, x, lengths):
        mask = length_to_mask(lengths)[:, :, None]
        x = x.masked_fill(~mask, 0)
        n_elements = mask.sum()
        m = x.sum((0, 1), keepdim=True).div_(n_elements)
        m_x_2 = (x*x).sum((0, 1), keepdim=True).div_(n_elements)
        v = m_x_2 - m*m
        self.means.lerp_(m, self.mom)
        self.vars.lerp_ (v, self.mom)
        return m,v

    def forward(self, x, lengths):
        if self.training:
            with torch.no_grad(): m,v = self.update_stats(x, lengths)
        else: m,v = self.means,self.vars
        x = (x-m) / (v+self.eps).sqrt()
        return x*self.mults + self.adds


class MaskedLayerNorm(nn.LayerNorm):
    def forward(self, x):
        # Compute cummulative summary statics along time axis
        N = torch.arange(
            start=1., end=x.shape[1]+1, device=x.device)[None, :, None]
        mean_x = torch.cumsum(x, 1) / N
        std_x = torch.sqrt(torch.cumsum((x - mean_x) ** 2, 1) / N + self.eps)

        return ((x - mean_x) / std_x) * self.weight + self.bias


class ReZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x1, x2):
        return x1 + self.resweight * x2


class PositionwiseLinear(nn.Module):
    """Convenience function for position-wise linear projection."""

    def __init__(self, d_in, d_out):
        """Position-wise linear layer.

        Applies a linear projection along the time axis. This assumes the time
        axis to be at the second last dimension of the input. Thus the input
        should be of shape [...x time points x features]

        Args:
            d_in: Input dimensionality
            d_out: Output dimensionality
        """
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_out, 1)  # position-wise

    def forward(self, x):
        """Apply linear projection to axis -2 of input x."""
        x = x.permute(1, 2, 0)
        x = self.w_1(x).permute(2, 0, 1)
        return x


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You
    Need".  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention
    is all you need. In Advances in Neural Information Processing Systems,
    pages 6000-6010. Users may modify or implement in a different way during
    application.

    This class is adapted from the pytorch source code.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model
            (default=2048).
        dropout: the dropout value (default=0.1).
        norm: Normalization to apply, one of 'layer' or 'rezero'.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", norm='layer'):
        super(TransformerEncoderLayer, self).__init__()
        if norm == 'layer':
            def get_residual():
                def residual(x1, x2):
                    return x1 + x2
                return residual

            def get_norm():
                return nn.LayerNorm(d_model)
        elif norm == 'rezero':
            def get_residual():
                return ReZero()
            def get_norm():
                return nn.Identity()
        else:
            raise ValueError('Invalid normalization: {}'.format(norm))

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = get_norm()
        self.norm2 = get_norm()
        self.residual1 = get_residual()
        self.residual2 = get_residual()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.residual1(src, self.dropout1(src2))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.residual2(src, self.dropout2(src2))
        src = self.norm2(src)
        return src

class AttentionModel(BaseModel):
    """Sequence to sequence model based on MultiHeadAttention."""

    def __init__(self, hparams):
        """AttentionModel.

        Args:
            d_in: Dimensionality of a single time point
            d_model: Dimensionality of the model
            ff_dim: Dimensionality of ff layers
            n_layers: Number of MultiHeadAttention layers
            n_heads: Number of attention heads
        """
        super().__init__(hparams)
        d_model = hparams.d_model
        n_layers = hparams.n_layers
        n_heads = hparams.n_heads
        ff_dim = hparams.ff_dim
        dropout = hparams.dropout
        norm = hparams.norm
        d_in = self._get_input_dim()
        self.layers = nn.ModuleList(
            [PositionwiseLinear(d_in, d_model)]
            + [
                TransformerEncoderLayer(
                    d_model, n_heads, ff_dim, dropout, norm=norm)
                for n in range(n_layers)
            ]
            + [PositionwiseLinear(d_model, 1)]
        )

    @property
    def transforms(self):
        parent_transforms = super().transforms
        parent_transforms.extend([
            PositionalEncoding(1, 500, 10),  # apply positional encoding
            to_observation_tuples            # mask nan with zero add indicator
        ])
        return parent_transforms


    def forward(self, x, lengths):
        """Apply attention model to input x."""
        # Invert mask as multi head attention ignores values which are true
        mask = ~length_to_mask(lengths)
        out = x.permute(1, 0, 2)
        for layer in self.layers:
            out = layer(out)
        return out.permute(1, 0, 2)

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure):
        n_warmup = 1000.
        if self.trainer.global_step < n_warmup:
            lr_scale = min(1., float(self.trainer.global_step + 1) / n_warmup)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        optimizer.step()
        optimizer.zero_grad()

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        # MODEL specific
        parser.opt_list(
            '--d-model', type=int, default=128,
            tunable=True, options=[128, 256, 512]
        )
        parser.opt_list(
            '--n-layers', type=int, default=1,
            tunable=True, options=[1, 2, 3, 4, 5]
        )
        parser.opt_list(
            '--n-heads', type=int, default=8,
            tunable=True, options=[4, 8, 16, 32]
        )
        parser.opt_list(
            '--ff-dim', type=int, default=32,
            tunable=True, options=[128, 256, 512, 1028]
        )
        parser.add_argument(
            '--norm', type=str, default='rezero', choices=['layer', 'rezero'])
        parser.opt_list(
            '--dropout', default=0.1, type=float,
            tunable=True, options=[0., 0.1, 0.2, 0.3, 0.4]
        )
        return parser
