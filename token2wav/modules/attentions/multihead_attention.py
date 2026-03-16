import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from collections import namedtuple

from ..commons.layers import Linear, Dropout
from .utils import get_slopes, apply_rotary_pos_emb


AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @torch.autocast(enabled = False, device_type="cuda")
    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.arange(t, device = self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs
    

class AlibiEmbedding(nn.Module):
    """ Symmetric version of Alibi"""
    def __init__(self, num_heads, max_position_embeddings=1024):
        super().__init__()
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.register_buffer("alibi", self.build_alibi_tensor())

    @property
    def device(self):
        return self.alibi.device

    def build_alibi_tensor(self):
        # For simplicity, compute the symmetric distance among all tokens,
        # thus the mask is important for causal modeling (Shan)
        context_position = torch.arange(self.max_position_embeddings)[:, None]
        memory_position = torch.arange(self.max_position_embeddings)[None, :]
        relative_position = memory_position - context_position 
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.num_heads, -1,-1)
        slopes = torch.Tensor(get_slopes(self.num_heads)) * -1
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        alibi = alibi.view(1, self.num_heads, self.max_position_embeddings, self.max_position_embeddings)
        return alibi

    @torch.autocast(enabled = False, device_type="cuda")
    def forward(self, L, S):
        max_len = max(L, S)
        # Update
        if max_len > self.max_position_embeddings:
            print(f"Updating alibi matrix for {self.device}")
            self.max_position_embeddings = max_len
            self.register_buffer("alibi", self.build_alibi_tensor())
        return self.alibi[:, :, :L, :S]
 

class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    def __init__(
            self,
            hidden_size: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            dropout: float = 0.0,
            max_position_embeddings: int = 4096,
            norm_layer: nn.Module = nn.LayerNorm,
            alibi_bias: bool = False,
            rotary_bias: bool = False,
            **kwargs
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, 'hidden_size should be divisible by num_heads'
        assert not( alibi_bias and rotary_bias), 'alibi_bias rotary_bias cannot be all True'
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_position_embeddings = max_position_embeddings
        self.alibi_bias = alibi_bias
        self.rotary_bias = rotary_bias 

        self.q_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = Dropout(attn_drop)
        self.o_proj = Linear(hidden_size, hidden_size)
        self.o_dropout = Dropout(dropout)

        self.cpu_config = AttentionConfig(True, True, True)
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            # TODO, should be True, False, False, but got error in my docker images
            self.cuda_config = AttentionConfig(True, True, True)
        else:
            self.cuda_config = AttentionConfig(False, True, True)

        if self.alibi_bias: 
            self.alibi = AlibiEmbedding(self.num_heads)

        if self.rotary_bias: 
            self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, q, k=None, v=None, mask=None):
        k = k or q
        v = v or q
        B, L, C = q.shape
        B, S, C = v.shape
        if mask is not None:
            if mask.ndim == 2: # [B, L]
                assert L == S
                mask = rearrange(mask, 'b j -> b 1 1 j')
                mask = mask.expand(-1, self.num_heads, L, -1)
            elif mask.ndim == 3: # [B, L, S]
                assert mask.size(1) == L and mask.size(2) == S
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)

        config = self.cuda_config if q.is_cuda else self.cpu_config
        attn_bias = torch.zeros(B, self.num_heads, L, S, dtype=q.dtype, device=q.device)

        # Apply alibi
        if self.alibi_bias:
            attn_bias += self.alibi(L, S)

        # Apply rotary
        if self.rotary_bias:
            if L == S:
                rotary_emb = self.rotary(L)
                q, k = map(lambda x: apply_rotary_pos_emb(rotary_emb, x), (q, k))
            else:
                q_rotary_emb = self.rotary(L)
                k_rotary_emb = self.rotary(S)
                q = apply_rotary_pos_emb(q_rotary_emb, q)
                k = apply_rotary_pos_emb(k_rotary_emb, k)

        if mask is not None:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))


        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.o_dropout(self.o_proj(out))
        return out


if __name__ == "__main__":
    from utils.util import exists, get_mask_from_lengths
    x = torch.rand(2,10,128)
    mask = get_mask_from_lengths(torch.LongTensor([10,2]))
    net = MultiHeadAttention(128, max_position_embeddings=12, num_heads=16)
    y = net(x, mask=mask)
