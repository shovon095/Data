python3 llama_interpret.py     --model_path ./checkpoints/llama_bird_ft/                --data_file   ./data/dev/dev.json                     --db_root    ./data/dev/dev_databases                 --max_new_tokens 128
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.28it/s]
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Loaded 1534 examples.
LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 221, in <module>
    main(args)
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 131, in main
    clause = clause_tags[o]
IndexError: list index out of range

