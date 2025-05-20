/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Loaded 1 examples.
Processed california_schools: exec=False
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 208, in <module>
    main(args)
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 185, in main
    X=np.stack(features[c]); y=np.array(labels[c])
IndexError: too many indices for tensor of dimension 2
