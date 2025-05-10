(base) shouvon@dxs4-DGX-Station:~/bioinform$ python train.py   --input     ./tms_concat.h5ad   --output    ./scvi_tms   --annotated ./TMS_scVI_integrated.h5ad
/home/shouvon/miniconda3/lib/python3.11/site-packages/docrep/decorators.py:43: SyntaxWarning: 'param_categorical_covariate_keys' is not a valid key!
  doc = func(self, args[0].__doc__, *args[1:], **kwargs)
/home/shouvon/miniconda3/lib/python3.11/site-packages/docrep/decorators.py:43: SyntaxWarning: 'param_continuous_covariate_keys' is not a valid key!
  doc = func(self, args[0].__doc__, *args[1:], **kwargs)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
/home/shouvon/miniconda3/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:425: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=39` in the `DataLoader` to improve performance.
/home/shouvon/miniconda3/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:425: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=39` in the `DataLoader` to improve performance.
Epoch 20/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [08:35<00:00, 25.77s/it, v_num=1, train_loss_step=9.58e+3, train_loss_epoch=1.03e+4]`Trainer.fit` stopped: `max_epochs=20` reached.
Epoch 20/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [08:35<00:00, 25.77s/it, v_num=1, train_loss_step=9.58e+3, train_loss_epoch=1.03e+4]

[train_scvi] Model saved to ./scvi_tms and annotated data to ./TMS_scVI_integrated.h5ad
