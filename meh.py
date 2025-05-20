
 python3 llama_interpret.py     --model_path ./checkpoints/llama_bird_ft/                --data_file   ./data/dev/dev.json                     --db_root    ./data/dev/dev_databases                 --max_new_tokens 128
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.29it/s]
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 215, in <module>
    main(args)
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 69, in main
    examples = [json.loads(l) for l in open(args.data_file)]
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 69, in <listcomp>
    examples = [json.loads(l) for l in open(args.data_file)]
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 2 column 1 (char 2)
