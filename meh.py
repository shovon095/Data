python no_adap.py   --input     5xFAD_matched_with_protocol.h5ad   --model_dir scvi_tms   --output    results/fad_no_adapter.h5ad
/home/shouvon/miniconda3/lib/python3.11/site-packages/docrep/decorators.py:43: SyntaxWarning: 'param_categorical_covariate_keys' is not a valid key!
  doc = func(self, args[0].__doc__, *args[1:], **kwargs)
/home/shouvon/miniconda3/lib/python3.11/site-packages/docrep/decorators.py:43: SyntaxWarning: 'param_continuous_covariate_keys' is not a valid key!
  doc = func(self, args[0].__doc__, *args[1:], **kwargs)
🧬 Dataset distribution:
dataset
WT    88296
Name: count, dtype: int64
INFO     File scvi_tms/model.pt already downloaded
/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_base_field.py:63: UserWarning: adata.X does not contain unnormalized count data. Are you sure this is what you want?
  self.validate_field(adata)
