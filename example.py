import torch
import time
import argparse
import soundfile
import os
from pathlib import Path

from transformers import Qwen2Config, WhisperConfig
from transformers import AutoConfig, AutoModel, LogitsProcessorList
from transformers.models.qwen2 import Qwen2TokenizerFast

from covoaudio.configuration_covo_audio import CovoAudioConfig
from covoaudio.modeling_covo_audio import CovoAudioForCausalLM, get_dialog_prompt, WindowedRepetitionPenaltyLogitsProcessor
from token2wav.decode import init_model, decode


def process_output(tokens, tokenizer, out_dir, mode="a2ta", round=0):
    filtered = tokens[tokens < len(tokenizer)]
    decoded = tokenizer.decode(filtered, skip_special_tokens=True)
    print(f"Decoded text: {decoded}")
    if mode != "a2ta":
        return
    
    projected_tokens = (tokens - len(tokenizer)).cpu().numpy().tolist()
    audio_tokens = [token for token in projected_tokens if 0 <= token <= 16384]
    decode_start = time.time()
    decode_res = decode(
        llm_tokens=audio_tokens,
        prompt_dir=prompt_dir,
        model=decode_model,
        config=decode_config,
    )
    decode_end = time.time()
    print(f"Audio decoding time: {decode_end - decode_start:.2f} seconds")

    soundfile.write(
        f"{out_dir}/turn_{round}.wav",
        decode_res["wav"],
        decode_res["sample_rate"]
    )


def get_parser():
    parser = argparse.ArgumentParser("covo-audio server")
    parser.add_argument("--model_dir", type=str, required=True, help="directory of the covo-audio model")
    parser.add_argument("--mode", type=str, default="a2ta", help="generation mode: a2t (audio to text), a2ta (audio to text and audio)")
    # below args are required for audio decoding when mode is a2ta
    parser.add_argument("--decode_model_config", type=str, default=None, help="model config file")
    parser.add_argument("--decode_load_path", type=str, default=None, help="pretrained model checkpoint path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="directory for prompt wavs")
    
    return parser


if  __name__ == "__main__":
    AutoConfig.register("covo_audio", CovoAudioConfig)
    AutoModel.register(CovoAudioConfig, CovoAudioForCausalLM)
    CovoAudioConfig.register_for_auto_class("AutoConfig")
    CovoAudioForCausalLM.register_for_auto_class("AutoModel")

    parser = get_parser()
    args = parser.parse_args()

    print("Start loading the Covo-Audio model...")
    model_dir = args.model_dir
    # model = CovoAudioForCausalLM.from_pretrained(model_dir)
    init_time = time.time()
    config = CovoAudioConfig.from_pretrained(model_dir)
    model = AutoModel.from_config(config)
    model = model.to("cuda:0")
    model.eval()
    init_over = time.time()
    print(f"Model initialization time: {init_over - init_time:.2f} seconds")
    
    from safetensors.torch import load_file
    files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    start = time.time()
    for f in sorted(files):
        state_dict = load_file(os.path.join(model_dir, f))
        model.load_state_dict(state_dict, strict=False) 
    end = time.time()
    print(f"Weight loading time: {end - start:.2f} seconds")
    
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_dir)

    mode = args.mode  # audio to text and audio
    if mode == "a2ta":
        eos_token_id = tokenizer.encode("<|im_end|>")[0]
    elif mode == "a2t":
        eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    else:
        print(f"Mode {mode} not supported yet")
        exit(0)

    # Preparation for decoding audio
    out_dir = None
    if mode == "a2ta":
        global decode_model, decode_config, prompt_dir
        prompt_dir = args.prompt_dir
        print("Initialize decode model...")
        decode_model, decode_config = init_model(args, decode_device="cuda:1")
        
        session_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) 
        out_dir = f"./Decoded_audios/{session_id}"
        os.makedirs(out_dir, exist_ok=True)
    
    # set up logits processor to ensure the quality of audio generation
    logits_processor = LogitsProcessorList()
    logits_processor.append(
        WindowedRepetitionPenaltyLogitsProcessor(penalty=1.05, window_size=8)
    )

    # prepare the inputs 
    wavs, input_ids, attention_mask = get_dialog_prompt(
        audio="./testdata/003000298.wav", 
        tokenizer=tokenizer, device="cuda:0"
    )

    start_time = time.time()
    input_len = input_ids.shape[1]
    result = model.generate(
        input_ids=input_ids, 
        wavs=wavs,  
        attention_mask=attention_mask,
        max_new_tokens=2048,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        logits_processor=logits_processor,
        repetition_penalty=1.05,
    )
    end_time = time.time()
    print(f"Generation time (round 0): {end_time - start_time:.2f} seconds")
    
    seq = result.sequences
    history = result.past_key_values
    tokens = seq.squeeze(0)[input_len:] 
    process_output(tokens, tokenizer, out_dir, mode, round=0)


    # continue the interaction with history
    wavs, input_ids, attention_mask = get_dialog_prompt(
        audio="./testdata/003000297.wav", 
        tokenizer=tokenizer, device="cuda:0", first_round=False
    )
    # construct an all-ones mask that includes the history
    history_len = history.get_seq_length() 
    current_mask_len = attention_mask.shape[1]
    full_attention_mask = torch.ones(
        (1, history_len + current_mask_len), device=attention_mask.device
    )

    input_len = input_ids.shape[1]
    start_time = time.time()
    result = model.generate(
        input_ids=input_ids, 
        wavs=wavs, 
        attention_mask=full_attention_mask,
        max_new_tokens=2048,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        past_key_values=history, # use the history from previous round
        return_dict_in_generate=True,
        logits_processor=logits_processor,
        repetition_penalty=1.05,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        top_k=50,
    )
    end_time = time.time()
    print(f"\nGeneration time (round 1): {end_time - start_time:.2f} seconds")
    
    seq = result.sequences
    history = result.past_key_values
    tokens = seq.squeeze(0)[input_len:] 
    process_output(tokens, tokenizer, out_dir, mode, round=1)
 
