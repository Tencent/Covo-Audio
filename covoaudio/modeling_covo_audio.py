import torch
import torchaudio
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.models.qwen2 import Qwen2ForCausalLM
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from .configuration_covo_audio import CovoAudioConfig

from torch import nn
import numpy as np
import torch.nn.functional as F
import os
from functools import lru_cache
from typing import Optional, Union


@torch.no_grad()
def get_dialog_prompt(audio, tokenizer, device, first_round=True):
    begofcAUDIO_id, cAUDIO_id, endofcAUDIO_id = tokenizer.convert_tokens_to_ids(["<|begofcAUDIO|>", "<|cAUDIO|>", "<|endofcAUDIO|>"])

    wav, sr = torchaudio.load(audio)
    if wav.shape[0] == 2:  # stereo to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)

    # hyperparameters
    sample_rate = 24000
    pad_multiple = True
    multiple_of = 480
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sample_rate)

    hop_size = sample_rate // 100
    wav = wav[: len(wav) // hop_size * hop_size]
                           
    # pad wav
    if pad_multiple and multiple_of is not None:
        d = (wav.shape[0] + multiple_of - 1) // multiple_of * multiple_of - wav.shape[0]
        if d > 0:
            wav = F.pad(wav, (0, d), value=0)
    num_token = calc_seq_len(len(wav) * 100 // sample_rate)
    
    
    # first round dialog
    if first_round:
        sys_prompt = """你是"小腾"，英文名是"Covo"，由腾讯开发的AI助手。
1、请使用简洁、口语化的语言和用户聊天，你的态度积极、耐心，像一位值得信赖的朋友。
2、不要使用列表或编号，避免输出网址、表情符号和复杂的公式。
3、不评价竞争对手，不发表主观政治观点，针对色情类、政治类、恐怖类、歧视类、暴力类的用户问题，你要妥善应对潜在的安全风险，并给出幽默，情绪安抚以及安全的劝导。
请用文本和音频进行对话，交替生成5个文本token和15个音频token，音频部分使用发音人：default_female"""
        interleave_text = "<|begofcAUDIO|>" + "<|cAUDIO|>" * num_token + "<|endofcAUDIO|>"
        
        sys_prompt = "<|im_start|>system\n" + sys_prompt + "<|im_end|>\n"
        prompt = sys_prompt + "<|im_start|>user\n" + interleave_text + "<|im_end|>\n<|im_start|>assistant\n"
    # multi-round dialog
    else:
        interleave_text = "<|begofcAUDIO|>" + "<|cAUDIO|>" * num_token + "<|endofcAUDIO|>"
        prompt = "\n<|im_start|>user\n" + interleave_text + "<|im_end|>\n<|im_start|>assistant\n"

    
    text_inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
    input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask
    
    wav = wav.to(device)
    # long audio (>30s) processing support
    segment_length = 720000  # 30s * 24000Hz
    # calculate total number of segments
    total_length = wav.shape[0]
    num_segments = (total_length + segment_length - 1) // segment_length
    wav_segments = []
    # split into 30s segments and collect
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, total_length)
        # extract current segment
        segment = wav[start_idx:end_idx]
        wav_segments.append(segment)
    return wav_segments, input_ids, attention_mask


def sequence_mask(lengths, max_len=None, dtype=torch.bool):
    if max_len is None:
        max_len = lengths.max()
    #mask = ~(torch.ones((len(lengths), max_len)).to(lengths.device).cumsum(dim=1).t() > lengths).t()
    mask = ~(torch.ones((len(lengths), max_len)).to(lengths.device).cumsum(dim=1) > lengths.unsqueeze(1))
    mask = mask.to(dtype)
    return mask


def calc_seq_len(seq_len):
    strides = [2, 2, 2, 2]
    for s in strides:
        seq_len = (seq_len + s - 1) // s
    return seq_len


class DownsampleLayer(nn.Module):
    """
    Downsample layer with 1D convolution and linear layers.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=2048):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # -> (B, C, T)
        x = self.conv1d(x)     # -> (B, C, T // 2)
        x = x.transpose(1, 2)  # -> (B, T // 2, C)
        x = self.relu1(x)
        x = self.linear1(x)    # -> (B, T // 2, hidden_dim)
        x = self.relu2(x)
        x = self.linear2(x)    # -> (B, T // 2, output_dim)
        return x


class AudioAdapter(nn.Module):
    """
    Audio adapter with downsample layers.
    """
    def __init__(self, input_dim, output_dim, downsample=8):
        """
        Args:
            input_dim (int): input feature dimension (number of channels)
            output_dim (int): output feature dimension
            downsample (int): total downsampling factor, must be a power of 2
        """
        super(AudioAdapter, self).__init__()
        assert downsample % 2 == 0 and downsample >= 2, "downsample must be even"

        num_layers = downsample.bit_length() - 1  # calculate how many downsampling steps are needed to reach the target factor

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_dim = output_dim if is_last else input_dim
            layers.append(DownsampleLayer(in_dim, out_dim))
            in_dim = out_dim  

        self.downsample_layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (B, T, C),C=input_dim
        Returns:
            Tensor: shape (B, T // downsample, output_dim)
        """
        for layer in self.downsample_layers:
            x = layer(x)
        return x


# from openai-whisper
# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: torch.Tensor,
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    dtype = audio.dtype
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec.to(dtype)
    return log_spec


class WindowedRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float, window_size: int):
        self.penalty = penalty
        self.window_size = window_size

    def __call__(self, input_ids, scores):
        for batch_idx, input_seq in enumerate(input_ids):
            window = input_seq[-self.window_size:] if self.window_size > 0 else input_seq  # get last 'window_size' tokens
            for token_id in set(window.tolist()):
                if scores[batch_idx, token_id] < 0:
                    scores[batch_idx, token_id] *= self.penalty
                else:
                    scores[batch_idx, token_id] /= self.penalty
        return scores


class CovoAudioForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CovoAudioConfig
    
    def __init__(self, config: CovoAudioConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.llm = Qwen2ForCausalLM(config.llm_config)
        self.encoder = WhisperEncoder(config.encoder_config)
        self.audio_adapter = AudioAdapter(config.whisper_feats_dim, 
                                          config.llm_config.hidden_size, 
                                          config.adapter_downsample)
        
        self.post_init()
        
    #NOTE Force 'tie_weights' function to do nothing to 
    # avoid the memory sharing between input and output embeddings of llm
    def tie_weights(self, **kwargs):
        pass

    def audio_encoder(self, wavs, device):
        """
        Extract features from input waveform
        """
        # move resampler to the correct device
        resampler16k = torchaudio.transforms.Resample(24000, 16000).to(device)

        mel_features_list = []
        for wav in wavs:
            wav = resampler16k(wav)
            audio = pad_or_trim(wav)
            # [B, 80, 3000] 30s 100hz
            mel_features = log_mel_spectrogram(audio, n_mels=128).to(torch.bfloat16)
            mel_features_list.append(mel_features)
        mel_features = torch.stack(mel_features_list)

        feats = self.encoder(mel_features).last_hidden_state
        features = self.audio_adapter(feats)
        features = features.view(1, -1, features.shape[2])

        return features
    
    def forward(
        self, 
        input_ids=None, 
        inputs_embeds=None,
        wavs=None, 
        attention_mask=None, 
        past_key_values=None,
        labels=None, 
        position_ids=None,
        **kwargs
    ):
        outputs = self.llm(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            past_key_values=past_key_values, 
            labels=labels, 
            position_ids=position_ids, 
            **kwargs
        )
        return outputs
    
    def get_input_embeddings(self):
        """
        Return the model's input embeddings - required for GenerationMixin
        """
        return self.llm.get_input_embeddings()
    
    def get_output_embeddings(self):
        """
        Return the model's output embeddings - required for GenerationMixin
        """
        # return self.llm.get_output_embeddings()
        return self.llm.lm_head

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        attention_mask=None,
        **kwargs
    ):
        wavs = kwargs.get("wavs", None)
        is_first_iteration = kwargs.get("is_first_iteration", False)
        past_key_values = kwargs.get("past_key_values", None) 
        
        if is_first_iteration:      # First generation step, include audio processing
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            cAUDIO_id = 151666    # tokenizer.convert_tokens_to_ids("<|cAUDIO|>")
            audio_features = self.audio_encoder(wavs, inputs_embeds.device)
            feature_lengths = (input_ids == cAUDIO_id).sum(1)
            feature_seq_mask = sequence_mask(feature_lengths, max_len=audio_features.size(1), dtype=torch.bool)
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_features = audio_features[feature_seq_mask]
            
            audio_mask = input_ids == cAUDIO_id
            audio_mask = audio_mask.unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            return {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
        else:                       # We're in a generation step, no need to process audio again
            input_ids = input_ids[:, -1:]
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            return {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
    
    def _set_gradient_checkpointing(self, module, value=False):
        # For Qwen2
        if hasattr(self.llm, 'gradient_checkpointing'):
            self.llm.gradient_checkpointing = value

            # Add the missing _gradient_checkpointing_func method to Qwen2Model
            if value and not hasattr(self.llm, '_gradient_checkpointing_func'):
                def _gradient_checkpointing_func(module_to_run, *args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(module_to_run, *args, **kwargs)

                self.llm._gradient_checkpointing_func = _gradient_checkpointing_func

        # For custom encoder and adapter
        if hasattr(self.encoder, 'gradient_checkpointing'):
            self.encoder.gradient_checkpointing = value
        if hasattr(self.audio_adapter, 'gradient_checkpointing'):
            self.audio_adapter.gradient_checkpointing = value
