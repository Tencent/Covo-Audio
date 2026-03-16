import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm, remove_weight_norm

from .ops import get_padding


class Dropout(nn.Module):
    def __init__(
        self, 
        p: float = 0.5, 
        inplace: bool = False,
        force_drop: bool = False,
        **kwargs
    ):
        super(Dropout, self).__init__() 
        if p < 0. or p > 1.:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.force_drop = force_drop

    def forward(self, x, **kwargs):
        return F.dropout(
            x, 
            p=self.p, 
            training=True if self.force_drop else self.training,
            inplace=self.inplace
        )

    def extra_repr(self):
        return 'prob={}, inplace={}, force_drop={}'.format(
                self.p, self.inplace, self.force_drop)


class EmbeddingTable(nn.Embedding):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        pad_id=-1,
        **kwargs
    ):
        super().__init__(
            num_embeddings, 
            embedding_dim,
            **kwargs
        )
        
        nn.init.normal_(self.weight, 0.0, embedding_dim ** -0.5)
        self.pad_id = pad_id
        self.output_dim = embedding_dim

    def forward(self, x):
        if self.pad_id is not None:
            mask = x == self.pad_id
            x = x.masked_fill(mask, 0)
        outputs = super().forward(x)
        if self.pad_id is not None:
            outputs = outputs.masked_fill(mask.unsqueeze(-1), 0.)
        return outputs


class Linear(nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        w_init_gain: str = 'linear',
        activation = None,
        **kwargs
    ):
        super(Linear, self).__init__(
            in_channels,
            out_channels,
            bias=bias
        )

        self.activation = activation if activation is not None else nn.Identity()
        self.output_dim = out_channels
        if w_init_gain is not None:
            if isinstance(w_init_gain, str):
                gain = nn.init.calculate_gain(w_init_gain)
            else:
                gain = w_init_gain
            nn.init.xavier_uniform_(
                    self.weight, gain=gain)
        if bias:
            nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x, **kwargs):
        return self.activation(super(Linear, self).forward(x))


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        bias: bool = True,
        padding = None,
        causal: bool = False,
        bn: bool = False,
        activation = None,
        w_init_gain = None,
        input_transpose: bool = False,
        **kwargs
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias
        )

        self.in_channels = in_channels
        self.transpose = input_transpose
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        if w_init_gain is not None:
            nn.init.xavier_uniform_(
                self.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        outputs = self.activation(self.bn(super(Conv1d, self).forward(x)))
        return outputs.transpose(1, 2) if self.transpose else outputs

    def extra_repr(self):
        return '(settings): {}\n(causal): {}\n(input_transpose): {}'.format(
                super(Conv1d, self).extra_repr(), self.causal, self.transpose)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = 'zeros',
        causal: bool = False,
        input_transpose: bool = False,
        **kwargs
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, \
                    "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."

        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode
        )

        self.causal = causal
        self.stride = stride
        self.transpose = input_transpose

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        x = super(ConvTranspose1d, self).forward(x)
        if self.causal:
            x = x[:, :, :-self.stride]
        return x.transpose(1, 2) if self.transpose else x

    def extra_repr(self):
        return '(settings): {}\n(causal): {}\n(input_transpose): {}'.format(
                super(ConvTranspose1d, self).extra_repr(), self.causal, self.transpose)


class ConvPositionEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 31,
        groups: int = 16,
    ):
        super().__init__()
        self.conv = Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=groups,
            input_transpose=True,
            activation=nn.GELU())

    def forward(self, x, mask=None):

        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.)

        x = self.conv(x)
        if mask is not None:
            x = x.masked_fill(~mask, 0.)

        return x

