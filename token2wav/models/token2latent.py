import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

import random
import numpy as np
from typing import Optional

from ..utils.util import exists
from ..modules.commons.mask import get_mask_from_lengths, compute_random_span_mask, mask_data
from ..modules.commons.layers import EmbeddingTable, Linear, ConvTranspose1d, ConvPositionEmbed
from ..modules.commons.ops import eval_decorator

from .basic_model import BaseModel
from ..modules.flow_matching.helpers import BaseFlowMatchingHelper
from ..modules.dit.modules import TimestepEmbedder, FinalLayer

from ..modules.attentions.modules import TransformerBlock, Mlp
from ..modules.attentions.multihead_attention import MultiHeadAttention



class Token2latentFlowMatching(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.transformer.hidden_size
        self.token_input_dim = config.get("token_input_dim", self.model_dim)
        self.target_dim = config.z_dim

        self.helper = BaseFlowMatchingHelper(config.sigma)
        # speaker condition
        self.spkr_embed_dim = config.get("spkr_embed_dim", 512)
        self.cond_proj = Linear(self.model_dim+self.spkr_embed_dim, self.model_dim)
        self.spkr_mask_ratio = config.get("spkr_mask_ratio", 0.0)
        # token
        self.token_input_dim = self.config.get("token_input_dim", self.model_dim)
        self.token_pad_id = -1
        if config.upsample_factor > 1:
            self.token_proj = nn.Sequential(
                Linear(self.token_input_dim, self.model_dim, bias=True),
                ConvTranspose1d(
                    self.model_dim,
                    self.model_dim,
                    stride = config.upsample_factor,
                    kernel_size = config.upsample_factor * 2)
            )
        else:
            self.token_proj = Linear(self.token_input_dim, self.model_dim)
        # transformer input projection, 
        self.transformer_input_proj = Linear(
            self.model_dim + self.target_dim * 2,
            self.model_dim)
        self.bert_maskrate0 = getattr(config, "bert_mask_rate0", 0.7)
        self.bert_maskrate1 = getattr(config, "bert_mask_rate1", 1.0)
        self.random_maskrate = getattr(config, "random_maskrate", 0.3)
        self.cfg_droprate = getattr(config, "cfg_dropout", 0.2)

        # Sinusoidal positional embedding for time
        self.time_embedder = TimestepEmbedder(self.model_dim)
        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(
            hidden_size=self.model_dim,
            kernel_size=31,
            groups=16
        )
        # Build transformers
        self.blocks = nn.ModuleList()
        for _ in range(config.transformer.num_layers):
            attn_block = MultiHeadAttention(**config.transformer)
            ffn_block = Mlp(
                act_layer=lambda: nn.GELU(approximate="tanh"),
                **config.transformer)
            self.blocks.append(
                TransformerBlock(
                    attn_block,
                    ffn_block,
                    **config.transformer))
        self.output_layer = FinalLayer(config.transformer.hidden_size, self.target_dim)

    @property
    def use_spkr_embed(self):
        return True

    def cond_mask_spkr_embed(self, x, spkr_embed):
        b, device = x.size(0), x.device
        if not exists(spkr_embed):
            spkr_embed = torch.zeros(b, self.spkr_embed_dim, device=device, dtype=x.dtype)
        if self.spkr_mask_ratio > 0.0 and self.training:
            rand_mask = (torch.zeros(b).float().uniform_(0, 1) < self.spkr_mask_ratio).to(device)
            spkr_embed = mask_data(spkr_embed, rand_mask, masking_value=0.0)
        return spkr_embed

    def forward(
        self,
        token: torch.Tensor,
        token_lens: torch.Tensor,
        target: torch.Tensor,
        target_lens: torch.Tensor,
        spkr_embed: Optional[torch.Tensor] = None
    ):
        # tokens are already embeded
        b = token.size(0)
        # drop for cfg
        cfg_mask = (torch.zeros((b,)).float().uniform_(0, 1) < self.cfg_droprate).to(token.device)
        token = self.token_proj(token)
        token = mask_data(token, cfg_mask, masking_value=0.0)  # ensure zero for masked position
        ge = self.cond_mask_spkr_embed(token, spkr_embed)
        # Prepare CFM
        xt, ut, times = self.helper.compute_xt_ut(target)

        # Prepare Mask
        span_mask = compute_random_span_mask(
            target,
            mask_ratio_range=(self.bert_maskrate0, self.bert_maskrate1),
            x_lens=target_lens,
            tail_mask=True)  #[b,t,d]
        # cfg mask not conducted on latent by default
        latent_cond = mask_data(target, span_mask, masking_value=0)
        # prepare input and mask for transformers
        attn_mask = get_mask_from_lengths(target_lens, max_len=target.size(1))
        inputs = torch.cat([token, latent_cond, xt], dim = -1)

        outputs = self.vectorfield_forward(inputs, times, attn_mask, g_cond=ge)
        res = {
            "pred": outputs,
            "target": ut,
            "pred_lens": target_lens,
            "cond_mask": span_mask.unsqueeze(2)
        }
        return res

    def vectorfield_forward(self, inputs, times, self_attn_mask, g_cond=None):

        # Apply sinusoidal positional embedding 
        t = self.time_embedder(times)
        # c = t if g_cond is None else t + g_cond
        if g_cond is None:
            c = t
        else:
            cond_inp = torch.cat([t, g_cond], dim=-1)
            c = self.cond_proj(cond_inp)

        # Apply convolutional positional encoder
        ut = self.transformer_input_proj(inputs)
        ut = self.conv_embed(ut, mask = self_attn_mask) + ut

        # Run through transformer
        for block in self.blocks:
            ut = block(ut, c, mask=self_attn_mask)
        # Apply transformer output layer
        ut = self.output_layer(ut, c)
        return ut

    @eval_decorator
    @torch.no_grad()
    def inference(
        self,
        *,
        token: torch.Tensor,
        prefix_target: Optional[torch.Tensor] = None,
        spkr_embed: Optional[torch.Tensor] = None,
        s_steps: Optional[int] = 10,
        cfg_alpha: Optional[float] = 2.,
        rescale_logits: bool = False,
        **kwargs
    ):
        b, device = token.size(0), token.device
        assert b == 1, f"Only support batch_size == 1 when inference"

        token = self.token_proj(token)
        ge = self.cond_mask_spkr_embed(token, spkr_embed)
        # prepare latent and mask
        tgt_lens = token.size(1)
        latent_cond = torch.zeros(b, tgt_lens, self.target_dim).to(device)
        if prefix_target is not None:
            latent_cond[:,:prefix_target.size(1),:] = prefix_target
        # Restore audio
        sample, trajectory = self.sample(
            tokens=token,
            audio=latent_cond,
            steps=s_steps,
            alpha=cfg_alpha,
            g_cond=ge,
            rescale_logits=rescale_logits)
        return sample

    def sample(self, tokens, audio, steps, alpha = None, g_cond=None, rescale_logits=False):

        # Create noise
        noise = torch.randn([audio.size(0),audio.shape[1],self.target_dim],device=audio.device)

        # Create time interpolation
        times = torch.linspace(0, 1, steps, device = audio.device)

        def solver(t, z):

            # If alpha is not provided
            if alpha is None:
                output = torch.cat([tokens, audio, z], dim = -1)
                return self.vectorfield_forward(inputs=output,
                                 times=t.unsqueeze(0),
                                 self_attn_mask=None,
                                 g_cond=g_cond)
            # If alpha is provided - zero out tokens and audio and mix together
            tokens_empty = torch.zeros(*audio.shape[:2], self.model_dim, device=tokens.device, dtype=tokens.dtype)
            audio_empty = audio
            # Mix together
            tokens_t = torch.cat([tokens_empty, tokens], dim = 0)
            audio_t = torch.cat([audio_empty, audio], dim = 0)
            audio_noizy_t = torch.cat([z, z], dim = 0) # Just double it
            t_t = torch.stack([t, t], dim = 0) # Just double it
            c = g_cond
            if g_cond is not None:
                c = torch.cat([g_cond, g_cond], dim=0)
            output = torch.cat([tokens_t, audio_t, audio_noizy_t], dim = -1)
            # Inference
            predicted_mix = self.vectorfield_forward(
                inputs=output,
                times = t_t,
                self_attn_mask = None,
                g_cond=c
            )
            predicted_conditioned = predicted_mix[1].unsqueeze(0)
            predicted_unconditioned = predicted_mix[0].unsqueeze(0)

            # CFG prediction
            prediction = predicted_conditioned + (predicted_conditioned - predicted_unconditioned) * alpha
            prediction_rescaled = predicted_conditioned.std() * (prediction / prediction.std())
            if rescale_logits:
                return prediction_rescaled
            else:
                return prediction

        trajectory = odeint(solver, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')
        return trajectory[-1], trajectory


class Token2latentFlowMatchingWithEmbed(Token2latentFlowMatching):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.vocab_size = config.token_vocab_size
        self.token_embedding = EmbeddingTable(
            num_embeddings = self.vocab_size,
            embedding_dim = self.token_input_dim,
            pad_id = self.token_pad_id
        )

    def forward(
        self,
        token: torch.Tensor,
        token_lens: torch.Tensor,
        target: torch.Tensor,
        target_lens: torch.Tensor,
        spkr_embed: Optional[torch.Tensor] = None
    ):
        token = self.token_embedding(token)
        return super().forward(
            token = token,
            token_lens = token_lens,
            target = target,
            target_lens = target_lens,
            spkr_embed = spkr_embed
        )
    
    def inference(self, *, token, prefix_target = None, spkr_embed = None, s_steps = 10, cfg_alpha = 2, rescale_logits = False, **kwargs):
        token = self.token_embedding(token)
        return super().inference(token = token, prefix_target = prefix_target, spkr_embed = spkr_embed, s_steps = s_steps, cfg_alpha = cfg_alpha, rescale_logits = rescale_logits, **kwargs)
