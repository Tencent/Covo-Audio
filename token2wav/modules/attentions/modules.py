import math
import torch
import torch.nn as nn

from .utils import modulate
from ..commons.layers import Linear, Dropout
from .multihead_attention import MultiHeadAttention


""" ref: github.com/facebookresearch/DiT"""
class Mlp(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            ffn_hidden_size=4096, 
            act_layer=nn.GELU, 
            dropout=0.,
            **kwargs
    ):
        super().__init__()
        self.fc1 = Linear(hidden_size, ffn_hidden_size)
        self.act = act_layer()
        self.fc2 = Linear(ffn_hidden_size, hidden_size)
        self.drop = Dropout(dropout)

    def forward(self, x, **kwargs):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = Linear(hidden_size, output_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class TransformerBlock(nn.Module):
    """Conditional transformer block"""
    def __init__(
            self, 
            attention: nn.Module,
            ffn: nn.Module,
            hidden_size: int = 1024,
            modulation: bool = False,
            eps: float = 1e-6,
            **kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=not modulation, eps=eps)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=not modulation, eps=eps)
        self.attn = attention # Attention block instance
        self.ffn = ffn # Feed-forward block instance
        self.modulation = modulation
        if modulation:
            self.modulation_layer = nn.Sequential(
                nn.SiLU(),
                Linear(hidden_size, 6 * hidden_size, bias=True)
            )
            # Zero-init from DiT
            nn.init.constant_(self.modulation_layer[-1].weight, 0.)
            nn.init.constant_(self.modulation_layer[-1].bias, 0.)

    def forward(self, x, condition=None, mask=None):
        if condition is None:
            assert not self.modulation, "Without global condition, must set modulation to False"
        else:
            assert self.modulation, "With global condition, must set modulation to True"
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation_layer(condition).chunk(6, dim=1)

        # Attention forward
        if condition is not None:
            x = x + gate_attn.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_attn, scale_attn), mask=mask)
        else:
            x = x + self.attn(self.norm1(x), mask=mask)

        # FFN forward
        if condition is not None:
            x = x + gate_ffn.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        else:
            x = x + self.ffn(self.norm2(x), mask=mask)
        return x


if __name__ == "__main__":
    from utils.util import get_mask_from_lengths
    net = TransformerBlock(128, ffn_hidden_size=256)
    x = torch.rand(2,10,128)
    t = torch.rand(2)
    mask = get_mask_from_lengths(torch.LongTensor([10,2]))
    y = net(x, c=None, mask=mask)
