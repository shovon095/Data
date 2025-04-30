/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_base_field.py:63: UserWarning: adata.X does not contain unnormalized count data. Are you sure this is what you want?
  self.validate_field(adata)
Traceback (most recent call last):
  File "/home/shouvon/bioinform/./map_fad.py", line 58, in <module>
    main(args)
  File "/home/shouvon/bioinform/./map_fad.py", line 15, in main
    model = scvi.model.SCVI.load_query_data(ad, args.model_dir)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/model/base/_archesmixin.py", line 127, in load_query_data
    setup_method(
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/model/_scvi.py", line 223, in setup_anndata
    adata_manager.register_fields(adata, **kwargs)
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/_manager.py", line 185, in register_fields
    self._add_field(
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/_manager.py", line 220, in _add_field
    field_registry[_constants._STATE_REGISTRY_KEY] = field.transfer_field(
                                                     ^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_dataframe_field.py", line 210, in transfer_field
    self.validate_field(adata_target)
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_dataframe_field.py", line 179, in validate_field
    raise KeyError(f"{self._original_attr_key} not found in adata.{self.attr_name}.")
KeyError: 'protocol not found in adata.obs.'
