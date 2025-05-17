drwxrwxr-x  3 shouvon shouvon       4096 Dec 19  2023 __MACOSX
drwxr-xr-x 71 shouvon shouvon       4096 May 17 01:25 train_databases
-rw-r--r--  1 shouvon shouvon 9347158408 Dec 19  2023 train_databases.zip
-rw-rw-r--  1 shouvon shouvon    1733605 Dec 19  2023 train_gold.sql
-rw-rw-r--  1 shouvon shouvon    4321661 Dec 19  2023 train.json
-rw-rw-r--  1 shouvon shouvon     726087 Dec 19  2023 train_tables.json
(gpt) shouvon@dgx3-DGX-Station:~/DAMO-ConvAI/bird/llm/data/train$ cd ..
(gpt) shouvon@dgx3-DGX-Station:~/DAMO-ConvAI/bird/llm/data$ cd ..
(gpt) shouvon@dgx3-DGX-Station:~/DAMO-ConvAI/bird/llm$ python3 llama.py     --do_train     --train_path  ./data/train/train.json     --db_root_path ./data/train/train_databases/     --engine meta-llama/Llama-2-7b-hf     --output_dir checkpoints/llama_bird_ft     --num_train_epochs 3     --per_device_train_batch_size 2     --learning_rate 3e-4
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.37s/it]
/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Detected kernel version 4.15.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
Traceback (most recent call last):
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 284, in <module>
    main()
  File "/home/shouvon/DAMO-ConvAI/bird/llm/llama.py", line 230, in main
    trainer.train()
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 1859, in train
    return inner_training_loop(
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 1888, in _inner_training_loop
    train_dataloader = self.get_train_dataloader()
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 874, in get_train_dataloader
    dataloader_params["sampler"] = self._get_train_sampler()
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/transformers/trainer.py", line 844, in _get_train_sampler
    return RandomSampler(self.train_dataset)
  File "/home/shouvon/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/utils/data/sampler.py", line 143, in __init__
    raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")
ValueError: num_samples should be a positive integer value, but got num_samples=0

./
address/
airline/
app_store/
authors/
beer_factory/
bike_share_1/
book_publishing_company/
books/
car_retails/
cars/
chicago_crime/
citeseer/
codebase_comments/
coinmarketcap/
college_completion/
computer_student/
cookbook/
craftbeer/
cs_semester/
disney/
donor/
european_football_1/
food_inspection/
food_inspection_2/
genes/
hockey/
human_resources/
ice_hockey_draft/
image_and_language/
language_corpus/
