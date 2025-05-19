Detected kernel version 4.15.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
  0%|                                                                                                                                                                                      | 0/441 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/utils/checkpoint.py:90: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/utils/checkpoint.py:90: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/utils/checkpoint.py:90: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/utils/checkpoint.py:90: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 171, in <module>
    main()
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 147, in main
    trainer.train()
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 1859, in train
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 171, in <module>
    return inner_training_loop(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 2203, in _inner_training_loop
    main()
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 147, in main
    trainer.train()
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 1859, in train
    tr_loss_step = self.training_step(model, inputs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 3147, in training_step
    return inner_training_loop(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 2203, in _inner_training_loop
    self.accelerator.backward(loss)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/accelerate/accelerator.py", line 2242, in backward
    tr_loss_step = self.training_step(model, inputs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 3147, in training_step
    self.scaler.scale(loss).backward(**kwargs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/_tensor.py", line 522, in backward
    torch.autograd.backward(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/autograd/__init__.py", line 266, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    RuntimeErrorself.accelerator.backward(loss):
element 0 of tensors does not require grad and does not have a grad_fn
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/accelerate/accelerator.py", line 2242, in backward
    self.scaler.scale(loss).backward(**kwargs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/_tensor.py", line 522, in backward
    torch.autograd.backward(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/autograd/__init__.py", line 266, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 171, in <module>
    main()
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 147, in main
    trainer.train()
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 1859, in train
    return inner_training_loop(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 2203, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 3147, in training_step
    self.accelerator.backward(loss)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/accelerate/accelerator.py", line 2242, in backward
    self.scaler.scale(loss).backward(**kwargs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/_tensor.py", line 522, in backward
    torch.autograd.backward(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/autograd/__init__.py", line 266, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 171, in <module>
    main()
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 147, in main
    trainer.train()
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 1859, in train
    return inner_training_loop(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 2203, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 3147, in training_step
    self.accelerator.backward(loss)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/accelerate/accelerator.py", line 2242, in backward
    self.scaler.scale(loss).backward(**kwargs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/_tensor.py", line 522, in backward
    torch.autograd.backward(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/autograd/__init__.py", line 266, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
  0%|                                                                                                                                                                                      | 0/441 [00:01<?, ?it/s]
[2025-05-18 23:10:36,249] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: 1) local_rank: 0 (pid: 15119) of binary: /home/shouvon/anaconda3/envs/gpt/bin/python
Traceback (most recent call last):
  File "/home/shouvon/anaconda3/envs/gpt/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 347, in wrapper
    return f(*args, **kwargs)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/distributed/run.py", line 812, in main
    run(args)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/distributed/run.py", line 803, in run
    elastic_launch(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 135, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 268, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
llama.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2025-05-18_23:10:36
  host      : dgx3-DGX-Station
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 15120)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2025-05-18_23:10:36
  host      : dgx3-DGX-Station
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 15121)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2025-05-18_23:10:36
  host      : dgx3-DGX-Station
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 15122)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-05-18_23:10:36
  host      : dgx3-DGX-Station
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 15119)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
