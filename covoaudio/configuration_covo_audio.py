from typing import Optional

from transformers import Qwen2Config, WhisperConfig
from transformers.configuration_utils import PretrainedConfig


class CovoAudioConfig(PretrainedConfig):
    model_type = "covo_audio"
    sub_configs = {"llm_config": Qwen2Config, "encoder_config": WhisperConfig} # type: ignore
    has_no_defaults_at_init = True
    def __init__(self,
                 llm_config:Optional[Qwen2Config]=None,
                 encoder_config:Optional[WhisperConfig]=None,
                 audio_token_index=151671,
                 adapter_downsample=8,
                 **kwargs):

        if llm_config is None:
            llm_config = Qwen2Config(
                architectures=[
                    "Qwen2ForCausalLM"
                ],
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151643,
                hidden_act="silu",
                hidden_size=3584,
                initializer_range=0.02,
                intermediate_size=18944,
                max_position_embeddings=131072,
                max_window_layers=28,
                model_type="qwen2",
                num_attention_heads=28,
                num_hidden_layers=28,
                num_key_value_heads=4,
                rms_norm_eps=1e-06,
                rope_scaling=None,
                rope_theta=1000000.0,
                sliding_window=131072,
                torch_dtype="bfloat16",
                use_cache=True,
                use_mrope=False,
                use_sliding_window=False,
                vocab_size=168055
            )
        if encoder_config is None:
            encoder_config = WhisperConfig(
                _name_or_path="openai/whisper-large-v3",
                activation_dropout=0.0,
                activation_function="gelu",
                apply_spec_augment=False,
                architectures=[
                    "WhisperForConditionalGeneration"
                ],
                attention_dropout=0.0,
                begin_suppress_tokens=[
                    220,
                    50257
                ],
                bos_token_id=50257,
                classifier_proj_size=256,
                d_model=1280,
                decoder_attention_heads=20,
                decoder_ffn_dim=5120,
                decoder_layerdrop=0.0,
                decoder_layers=32,
                decoder_start_token_id=50258,
                dropout=0.0,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                eos_token_id=50257,
                init_std=0.02,
                mask_feature_length=10,
                mask_feature_min_masks=0,
                mask_feature_prob=0.0,
                mask_time_length=10,
                mask_time_min_masks=2,
                mask_time_prob=0.05,
                max_length=448,
                max_source_positions=1500,
                max_target_positions=448,
                median_filter_width=7,
                model_type="whisper",
                num_hidden_layers=32,
                num_mel_bins=128,
                scale_embedding=False,
                torch_dtype="float16",
                use_cache=True,
                use_weighted_layer_sum=False,
                vocab_size=51866
            )

        self.audio_token_index = audio_token_index
        self.adapter_downsample = adapter_downsample
        self.llm_config = llm_config
        self.encoder_config = encoder_config
        self.whisper_feats_dim = encoder_config.d_model
        
        if "dtype" not in kwargs:
            kwargs["dtype"] = "bfloat16"
        self.dtype = kwargs["dtype"]

        super().__init__(**kwargs)
    
    @property
    def num_hidden_layers(self):
        return self.llm_config.num_hidden_layers
    
    @property
    def hidden_size(self):
        return self.llm_config.hidden_size
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary, ensuring nested
        PretrainedConfig objects are serialized via their own to_dict().
        """
        output = super().to_dict()
        # replace nested config objects with their dict representation
        if hasattr(self, "llm_config") and isinstance(self.llm_config, PretrainedConfig):
            output["llm_config"] = self.llm_config.to_dict()
            output["_llm_config_type"] = getattr(self.llm_config, "model_type", None)
        if hasattr(self, "encoder_config") and isinstance(self.encoder_config, PretrainedConfig):
            output["encoder_config"] = self.encoder_config.to_dict()
            output["_encoder_config_type"] = getattr(self.encoder_config, "model_type", None)
        
        return output

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        """Create an CovoAudioConfig from a dict, reconstructing nested config
        objects (llm_config, encoder_config) using the classes declared in
        `sub_configs` if available.
        """
        # Make a shallow copy to avoid mutating input
        data = dict(config_dict)

        llm_conf = None
        enc_conf = None

        if "llm_config" in data and data["llm_config"] is not None:
            llm_cls = cls.sub_configs.get("llm_config") if hasattr(cls, "sub_configs") else None
            if llm_cls is not None:
                # use the sub-config class to reconstruct
                llm_conf = llm_cls.from_dict(data.pop("llm_config"))
            else:
                # fallback to raw dict
                llm_conf = data.pop("llm_config")

        if "encoder_config" in data and data["encoder_config"] is not None:
            enc_cls = cls.sub_configs.get("encoder_config") if hasattr(cls, "sub_configs") else None
            if enc_cls is not None:
                enc_conf = enc_cls.from_dict(data.pop("encoder_config"))
            else:
                enc_conf = data.pop("encoder_config")
                # ensure HF-compatible fields reflect the underlying decoder (LLM）

        # remove internal helper keys if present
        data.pop("_llm_config_type", None)
        data.pop("_encoder_config_type", None)

        # now construct instance using reconstructed nested configs
        return cls(llm_config=llm_conf, encoder_config=enc_conf, **data)
