decorations.
  plt.tight_layout()
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 225, in <module>
    main(args)
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama_interpret.py", line 186, in main
    lp_full = model(input_ids=inputs.input_ids, labels=target_ids).loss.item() * target_ids.size(1)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py", line 1241, in forward
    loss = loss_fct(shift_logits, shift_labels)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/nn/modules/loss.py", line 1179, in forward
    return F.cross_entropy(input, target, weight=self.weight,
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/nn/functional.py", line 3059, in cross_entropy
    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
ValueError: Expected input batch_size (38) to match target batch_size (129).
