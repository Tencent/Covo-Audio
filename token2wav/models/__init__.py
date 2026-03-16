from .audio_decoder import Token2WavDecoder

def get_tokenizer_wrapper(model_name, config, **kwargs):
    return eval(model_name)(config, **kwargs)
