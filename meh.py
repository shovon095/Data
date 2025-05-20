n 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Loaded 10 examples.
Processed california_schools: exec=False
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 198, in <module>
    main(args)
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 133, in main
    features["table"].append(v);  labels["table"].append(int(tok_str in tbls))
IndexError: too many indices for tensor of dimension 2
