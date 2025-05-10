home/shouvon/miniconda3/lib/python3.11/site-packages/docrep/decorators.py:43: SyntaxWarning: 'param_categorical_covariate_keys' is not a valid key!
  doc = func(self, args[0].__doc__, *args[1:], **kwargs)
/home/shouvon/miniconda3/lib/python3.11/site-packages/docrep/decorators.py:43: SyntaxWarning: 'param_continuous_covariate_keys' is not a valid key!
  doc = func(self, args[0].__doc__, *args[1:], **kwargs)
INFO     File scvi_tms/model.pt already downloaded
/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_base_field.py:63: UserWarning: adata.X does not contain unnormalized count data. Are you sure this is what you want?
  self.validate_field(adata)
Traceback (most recent call last):
  File "/home/shouvon/bioinform/no_adap.py", line 47, in <module>
    main(p.parse_args())
  File "/home/shouvon/bioinform/no_adap.py", line 23, in main
    ad.obsm["X_scVI_noAdapter"] = model.get_latent_representation()
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/model/base/_vaemixin.py", line 279, in get_latent_representation
    self._check_if_trained(warn=False)
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/model/base/_base_model.py", line 490, in _check_if_trained
    raise RuntimeError(message)
RuntimeError: Trying to query inferred values from an untrained model. Please train the model first.
python train.py   --input     ./tms_concat.h5ad   --output    ./scvi_tms   --annotated ./TMS_scVI_integrated.h5ad
