import sys, os
import json5, json
import torch
import numpy as np
import torchaudio

from .models import get_tokenizer_wrapper
from .utils.util import JsonHParams, load_ckpt



def decode(llm_tokens, prompt_dir, model, config):
    device = next(model.parameters()).device
    feature = {}
    feature["target_token"] = torch.LongTensor(llm_tokens).unsqueeze(0).to(device)
    model_sr = config.data.audio.sample_rate
    feature["sample_rate"] = model_sr
    
    with open(os.path.join(prompt_dir, "prompt_token.npy"), 'rb') as f:
        prompt_token = np.load(f)
    feature["prompt_token"] = torch.LongTensor(prompt_token).to(device)
    with open(os.path.join(prompt_dir, "prompt_latent.npy"), 'rb') as f:
        prompt_latent = np.load(f)
    feature["prompt_latent"] = torch.FloatTensor(prompt_latent).to(device)
    with open(os.path.join(prompt_dir, "speaker_embed.npy"), 'rb') as f:
        speaker_embed = np.load(f)
    feature["spkr_embed"] = torch.FloatTensor(speaker_embed).to(device)

    infer_conf = config.get("inference", {})

    audio_gen = model.inference(feature, **infer_conf)
    audio_gen = audio_gen.float().cpu().squeeze().numpy()
    return {"wav": audio_gen, "sample_rate": model_sr}


# --- Load decode model ---
def init_model(args, decode_device):
    with open(args.decode_model_config, 'r') as f:
        config = json5.loads(f.read())
    config = JsonHParams(**config)
    config.model.global_mean_var = os.path.join(os.path.dirname(args.decode_load_path), "global_mean_var.npy")
    model = get_tokenizer_wrapper(config.model.type, config.model)
    model = load_ckpt(model, args.decode_load_path)

    model = model.to(decode_device)
    model = model.eval()
    return model, config
