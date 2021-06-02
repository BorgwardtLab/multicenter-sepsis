"""Attention based model for sepsis."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.torch.models.base_model import BaseModel
from src.torch.torch_utils import PositionalEncoding, to_observation_tuples, to_observation_tuples_without_indicators


def get_subsequent_mask(seq, offset=0):
    """For masking out the subsequent info."""
    sz_b, len_s, n_features = seq.size()
    subsequent_mask = torch.triu(
        torch.ones(
            (len_s+offset, len_s+offset), device=seq.device, dtype=bool),
        diagonal=1
    )
    return subsequent_mask


def length_to_mask(length, max_len=None, dtype=None, offset=0):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or (length.max().item() + offset)
    mask = (
        torch.arange(max_len, device=length.device, dtype=length.dtype) \
        .expand(len(length), max_len)
        >= (length.unsqueeze(1) + offset)
    )
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class MaskedLayerNorm(nn.LayerNorm):
    def forward(self, x):
        # Compute cumulative summary statics along time axis
        N = torch.arange(
            start=1., end=x.shape[1]+1, device=x.device)[None, :, None]
        mean_x = torch.cumsum(x, 1) / N
        std_x = torch.sqrt(torch.cumsum((x - mean_x) ** 2, 1) / N + self.eps)

        return ((x - mean_x) / std_x) * self.weight + self.bias


class ReZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.resweight = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x1, x2):
        return x1 + self.resweight * x2


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
                 norm='layer'):
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
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = self.residual1(src, self.dropout1(src2))
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.residual2(src, self.dropout2(src2))
        src = self.norm2(src)
        return src


class AttentionModel(BaseModel):
    """Sequence to sequence model based on MultiHeadAttention."""

    def __init__(self, d_model, n_layers, n_heads, dropout, norm, indicators=False,
                 **kwargs):
        """AttentionModel.

        Args:
            d_model: Dimensionality of the model
            n_layers: Number of MultiHeadAttention layers
            n_heads: Number of attention heads
            indicators: flag if missingness indicators should be applied
        """
        super().__init__(**kwargs)
        ff_dim = 4*d_model # dimensionality of ff layers: hard-coded default
        self.to_observation_tuples = to_observation_tuples if indicators else to_observation_tuples_without_indicators 
        self.save_hyperparameters()
        d_statics, d_in = self._get_input_dims()
        if not self.hparams.ignore_statics:
            self.statics_embedding = nn.Linear(d_statics, d_model)
        self.layers = nn.ModuleList(
            [nn.Linear(d_in, d_model)]
            + [
                TransformerEncoderLayer(
                    d_model, n_heads, ff_dim, dropout, norm=norm)
                for n in range(n_layers)
            ]
            + [nn.Linear(d_model, 1)]
        )

    @property
    def transforms(self):
        parent_transforms = super().transforms
        parent_transforms.extend([
            PositionalEncoding(1, 500, 10),  # apply positional encoding
            self.to_observation_tuples            # mask nan with zero add indicator
        ])
        return parent_transforms

    def forward(self, x, lengths=None, statics=None):
        """Apply attention model to input x."""
        offset = 0 if statics is None or self.hparams.ignore_statics else 1
        # Invert mask as multi head attention ignores values which are true
        if lengths is None:
            # Assume we use nan to pad values. This helps when using shap for
            # explanations as it manipulates the input and automatically adds
            # noise to the lengths parameter (making it useless for us).
            not_all_nan = (~torch.all(torch.isnan(x), dim=-1)).long()
            # We want to find the last instance where not all inputs are nan.
            # We can do this by flipping the no_nan tensor along the time axis
            # and determining the position of the maximum. This should return
            # us the first maximum, i.e. the first time when (in the reversed
            # order) where the tensor does not contain only nans.
            # Strangely, torch.argmax and tensor.max do different things.
            lengths = not_all_nan.shape[1] - not_all_nan.flip(1).max(1).indices

        mask = length_to_mask(lengths, offset=offset, max_len=x.shape[1])
        future_mask = get_subsequent_mask(x, offset=offset)
        x = self.layers[0](x)
        if statics is not None and not self.hparams.ignore_statics:
            # prepend statics embedding
            embed_statics = self.statics_embedding(statics).unsqueeze(1)
            x = torch.cat([embed_statics, x], dim=1)
            ## alternative: append statics as channel dimensions
            #stacked_statics = statics.unsqueeze(1).repeat(1,x.shape[1],1)
            #x = torch.cat([x,stacked_statics], axis=-1)
            #x = self.statics_embedding(x)

        x = x.permute(1, 0, 2)
        for layer in self.layers[1:]:
            if isinstance(layer, TransformerEncoderLayer):
                x = layer(
                    x, src_key_padding_mask=mask, src_mask=future_mask)
            else:
                x = layer(x)
        x = x.permute(1, 0, 2)
        # Remove first element if statics are present
        x = x[:, offset:, :]
        return x

    # Somehow more recent versions of pytorch lightning don't work with the
    # below code. Need to check this out some other time.
    # def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure):
    #     n_warmup = 1000.
    #     if self.trainer.global_step < n_warmup:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / n_warmup)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams.learning_rate

    #     optimizer.step()
    #     optimizer.zero_grad()

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        # MODEL specific
        parser.add_argument(
            '--d_model', type=int, default=128,
            # tunable=True, options=[128, 256, 512]
        )
        parser.add_argument(
            '--n_layers', type=int, default=1,
            # tunable=True, options=[1, 2, 3, 4, 5]
        )
        parser.add_argument(
            '--n_heads', type=int, default=8,
            # tunable=True, options=[4, 8, 16, 32]
        )
        parser.add_argument(
            '--norm', type=str, default='rezero', choices=['layer', 'rezero'])
        parser.add_argument(
            '--dropout', default=0.1, type=float,
            # tunable=True, options=[0., 0.1, 0.2, 0.3, 0.4]
        )
        return parser
