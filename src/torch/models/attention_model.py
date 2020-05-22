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
        self.mom,self.eps = mom,eps
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


# pylint: disable=R0902,C0103,W0221
class MultiHeadAttention(nn.Module):
    """Multi head attention layer."""

    def __init__(self, d_model, n_heads, d_k, d_v, dropout, mask_future=True):
        """Multi head attention layer.

        Args:
            d_model: Embedding dimensionality
            n_heads: Number of attention heads
            qkd_v: Dimensionality of the q, k and v vectors
            mask_future: When predicting for a time point mask all future
                events
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.mask_future = mask_future

        self.w_q = nn.Linear(d_model, n_heads * d_k)
        self.w_k = nn.Linear(d_model, n_heads * d_k)
        self.w_v = nn.Linear(d_model, n_heads * d_v)
        self.attention = ScaledDotProductAttention(
            np.sqrt(d_k), dropout=dropout)
        self.w_o = nn.Linear(n_heads * d_v, d_model)

    def compute_mask(self, x):
        return get_subsequent_mask(x)

    def forward(self, x):
        """Apply multi head attention to input data x.

        Args:
            x: Tensor with [bs x time points x d_model]

        Returns:
            Transformed tensor

        """
        n_heads, d_k, d_v = self.n_heads, self.d_k, self.d_v
        bs, length, _ = x.size()

        q = self.w_q(x).view(bs, length, n_heads, d_k)
        k = self.w_k(x).view(bs, length, n_heads, d_k)
        v = self.w_v(x).view(bs, length, n_heads, d_v)

        # Combine heads and batch dimension.  Use chained continguous and view
        # to ensure we are not doing anything wrong.
        q = q.permute(2, 0, 1, 3).contiguous().view(bs*n_heads, length, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs*n_heads, length, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs*n_heads, length, d_v)

        mask = self.compute_mask(x) if self.mask_future else None

        z_flat, attention = self.attention(q, k, v, mask=mask)

        # Undo combination of heads and batch
        z = z_flat.view(n_heads, bs, length, d_v)
        z = z.permute(1, 2, 0, 3).contiguous()
        # Combine outputs from heads into single vector
        z = z.view(bs, length, n_heads*d_v)

        output = self.w_o(z)
        return output


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention from `Attention is all you need` paper."""

    def __init__(self, scale, dropout=0.1):
        """Scaled dot product attention.

        Args:
            scale: Scaling applied prior to softmax
            dropout: Fraction of values randomly dropped out from attention
        """
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """Apply dot product attention.

        All inputs are assumed to be 3d tensors.

        Args:
            q: Query [bs*n_heads x n_instances x dq]
            k: Keys [bs*n_heads x n_instances x dq]
            v: Values [bs*n_heads x n_instances x dq]
            mask: Broadcastable with [bs*n_heads x n_instances x n_instances]

        Returns: Output of attention operation and attention values
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention/self.scale

        if mask is not None:
            # -inf is ignored by the softmax function
            attention.masked_fill_(~mask, -np.inf)

        attention = self.softmax(attention)
        out = torch.bmm(self.dropout(attention), v)

        return out, attention


class PositionwiseFeedForward(nn.Module):
    """A two layer feed forward module."""

    def __init__(self, d_in, d_hid, dropout=0.1):
        """Position-wise feed-forward layer with residual.

        Applies two a two-layer neural network to the axis -2 of the input.

        Args:
            d_in: Dimensionality of input and output
            d_hid: Dimensionality of intermediate representation
            dropout: Fraction of nodes to randomly drop out
        """
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = BatchNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        residual = x
        output = x.transpose(-1, -2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(-1, -2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual, lengths)
        return output


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
        output = x.transpose(-1, -2)
        output = self.w_1(output).transpose(-1, -2)
        return output


class AttentionLayer(nn.Module):
    """Attention block with two MultiHeadAttention layers.

    One block consists of the following Sub-Layers:
        - MultiHeadAttention
        - LayerNorm
        - PositionwiseFeedForward
        - MultiHeadAttention
        - LayerNorm
    """

    def __init__(self, d_model, n_heads, qkv_dim, ff_hidden_dim=128,
                 dropout=0.1):
        """Attention block.

        Args:
            d_model: Dimensionality of the model
            n_heads: Number of attention heads
            qkd_v: Dimensionality of the q, k and v vectors
            ff_hidden_dim: Number of hidden units for PositionwiseFeedForward
        """
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttention(d_model, n_heads, qkv_dim, qkv_dim, dropout),
            BatchNorm(d_model),
            PositionwiseFeedForward(d_model, ff_hidden_dim),
            BatchNorm(d_model)
        ])

    def forward(self, x, lengths):
        """Apply this attention block to the input x."""
        out = x
        for layer in self.layers:
            if isinstance(layer, (BatchNorm, PositionwiseFeedForward)):
                out = layer(out, lengths)
            else:
                out = layer(out)
        return out


class AttentionModel(BaseModel):
    """Sequence to sequence model based on MultiHeadAttention."""

    def __init__(self, hparams):
        """AttentionModel.

        Args:
            d_in: Dimensionality of a single time point
            d_model: Dimensionality of the model
            n_layers: Number of MultiHeadAttention layers
            n_heads: Number of attention heads
            qkd_v: Dimensionality of the q, k and v vectors
        """
        super().__init__(hparams)
        d_model = hparams.d_model
        n_layers = hparams.n_layers
        n_heads = hparams.n_heads
        qkv_dim = hparams.qkv_dim
        dropout = hparams.dropout
        d_in = self._get_input_dim()
        self.layers = nn.ModuleList(
            [PositionwiseLinear(d_in, d_model)]
            + [AttentionLayer(d_model, n_heads, qkv_dim, dropout=dropout)
               for i in range(n_layers)]
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
        out = x
        for layer in self.layers:
            if isinstance(layer, (AttentionLayer, PositionwiseFeedForward)):
                out = layer(out, lengths)
            else:
                out = layer(out)
        return out

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
        parser.add_argument('--d-model', type=int, default=64)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--n-heads', type=int, default=8)
        parser.add_argument('--qkv-dim', type=int, default=32)
        parser.add_argument('--dropout', default=0.1, type=float)
        return parser
