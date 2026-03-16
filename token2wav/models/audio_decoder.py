import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from contextlib import nullcontext
from collections import OrderedDict

from ..utils.util import exists, load_ckpt
from ..modules.commons.mask import get_mask_from_lengths
from ..modules.commons.ops import eval_decorator

from .basic_model import BaseModel
from .latent2wav import BigVGANFlowVAE
from .token2latent import Token2latentFlowMatchingWithEmbed


class Token2WavDecoder(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.wavegan = BigVGANFlowVAE(config.wavegan)
        self.wavegan_hop_size = np.prod(config.wavegan.downsample_rates)
        self.global_mean_var = getattr(config, "global_mean_var", None)
        if self.global_mean_var is not None:
            print('load global mean and var for latents')
            mean_var_data = torch.from_numpy(np.load(self.global_mean_var)).float().squeeze()
            global_mean, global_var = mean_var_data.chunk(2, 0)
            self.register_buffer('global_mean', global_mean)
            self.register_buffer('global_var', global_var)
        else:
            print('no global mean and var for latents')

        # flow matching
        self.token2latent = Token2latentFlowMatchingWithEmbed(config.token2latent)

        # extra params
        self.upsample_factor = self.token2latent.config.get("upsample_factor", 1)
        self.wav_input_sr = config.get("wav_input_sr", 24000)

        # just avoid to train or save pretrained module
        self.trainable_module = ["wavegan", "token2latent"]

    def parameters(self, *args, **kwargs):
        for name in self.trainable_module:
            for param in self.get_submodule(name).parameters(*args, **kwargs):
                yield param

    def state_dict(self):
        param_dict = OrderedDict()
        for name in self.trainable_module:
            state = self.get_submodule(name).state_dict(prefix=f"{name}.")
            param_dict.update(state)
        return param_dict

    def load_state_dict(self, param_dict):
        for name in self.trainable_module:
            module_state = OrderedDict()
            name_len = len(name)
            for k, v in param_dict.items():
                if k.startswith(f"{name}."):
                    new_k = k[name_len+1:]
                    module_state[new_k] = v
            self.get_submodule(name).load_state_dict(module_state, strict=False)

    def train(self, mode=True):
        super().train(mode)
        self.wavegan.eval()
        
        return self

    


    @torch.no_grad()
    def preprocess_infer_data(self, data):
        sample_rate = data.get("sample_rate", None) or self.wav_input_sr
        zero_spkr = data.get("zero_spkr", False)
        wav_input = data.get("sample", None)
        
        token = data["target_token"]
        # token = F.embedding(token, self.tokenizer.vq._codebook.embed.data[0])
        prefix_target = None
        # prompt
        res_prefix_len = 0
        prompt_token = data.get("prompt_token", None)
        prompt_latent = data.get("prompt_latent", None)
        if exists(prompt_token):
            token = torch.cat([prompt_token, token], dim=1)
        if exists(prompt_latent):
            prefix_target = prompt_latent
            res_prefix_len = prompt_latent.shape[1]

        spkr_embed = None
        if not zero_spkr:
            spkr_embed = data.get("spkr_embed", None)
        res = {
            "token": token,
            "prefix_target": prefix_target,
            "spkr_embed": spkr_embed
        }
        return res, res_prefix_len

    @eval_decorator
    @torch.no_grad()
    def inference(self, data, **kwargs):
        infer_data, prefix_len = self.preprocess_infer_data(data)
        prefix_target = infer_data['prefix_target']
        res_latents = self.token2latent.inference(**infer_data, **kwargs)
        if self.global_mean_var is not None:
            res_latents = self.global_mean + res_latents * torch.sqrt(self.global_var)
            # replacd with gt prefix
            if exists(prefix_target):
                prefix_target = self.global_mean + prefix_target * torch.sqrt(self.global_var)
        if exists(prefix_target):
            res_latents[:, :prefix_target.shape[1]] = prefix_target
        res_latents = res_latents[:, prefix_len:]
        audio = self.wavegan.inference_from_latents(res_latents.transpose(1, 2), do_sample=False)
        return audio
