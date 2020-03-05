"""Attention based model for sepsis."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# pylint: disable=R0902,C0103,W0221
class MultiHeadAttention(nn.Module):
    """Multi head attention layer."""

    def __init__(self, d_model, n_heads, d_k, d_v, mask_future=False):
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
        self.attention = ScaledDotProductAttention(np.sqrt(d_k))
        self.w_o = nn.Linear(n_heads * d_v, d_model)

    def compute_mask(self, x):
        raise NotImplementedError()

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
            attention.masked_fill(mask, -np.inf)

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
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(-1, -2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(-1, -2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
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

    def __init__(self, d_model, n_heads, qkv_dim, ff_hidden_dim=128):
        """Attention block.

        Args:
            d_model: Dimensionality of the model
            n_heads: Number of attention heads
            qkd_v: Dimensionality of the q, k and v vectors
            ff_hidden_dim: Number of hidden units for PositionwiseFeedForward
        """
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttention(d_model, n_heads, qkv_dim, qkv_dim),
            nn.LayerNorm(d_model),
            PositionwiseFeedForward(d_model, ff_hidden_dim),
            nn.LayerNorm(d_model)
        ])

    def forward(self, x):
        """Apply this attention block to the input x."""
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class AttentionModel(nn.Module):
    """Sequence to sequence model based on MultiHeadAttention."""

    def __init__(self, d_in, d_model, n_layers, n_heads, qkv_dim):
        """AttentionModel.

        Args:
            d_in: Dimensionality of a single time point
            d_model: Dimensionality of the model
            n_layers: Number of MultiHeadAttention layers
            n_heads: Number of attention heads
            qkd_v: Dimensionality of the q, k and v vectors
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [PositionwiseLinear(d_in, d_model)]
            + [AttentionLayer(d_model, n_heads, qkv_dim)
               for i in range(n_layers)]
            + [PositionwiseLinear(d_model, 2)]
        )

    def forward(self, x):
        """Apply attention model to input x."""
        out = x
        for layer in self.layers:
            out = layer(out)
        return F.log_softmax(out, dim=-1)

    @classmethod
    def set_hyperparams(cls, kwargs={}):
        if kwargs['hypersearch']:
            raise NotImplementedError('Hyperparameter search not yet implemented!')
        else:
            defaults = {
                'd_model': 64,
                'n_layers': 1,
                'n_heads': 8,
                'qkv_dim': 32
            }
            kwargs.update(defaults)
            kwargs.pop('hypersearch')
            return cls(**kwargs) 
       



