model_dir="./covoaudio"
mode="a2ta"

prompt_dir="./token2wav/prompt"
decode_model_config="./token2wav/config.json"
decode_load_path="./covoaudio/token2wav/model.pt"

CUDA_VISIBLE_DEVICES=4,5 python example.py --model_dir $model_dir \
    --mode $mode \
    --prompt_dir $prompt_dir \
    --decode_model_config $decode_model_config \
    --decode_load_path $decode_load_path 
