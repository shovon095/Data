/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_base_field.py:63: UserWarning: adata.X does not contain unnormalized count data. Are you sure this is what you want?
  self.validate_field(adata)
Traceback (most recent call last):
  File "/home/shouvon/bioinform/no_adap.py", line 54, in <module>
    main(args)
  File "/home/shouvon/bioinform/no_adap.py", line 21, in main
    model = scvi.model.SCVI.load(args.model_dir, adata=ad)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/model/base/_base_model.py", line 731, in load
    getattr(cls, method_name)(adata, source_registry=registry, **registry[_SETUP_ARGS_KEY])
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/model/_scvi.py", line 223, in setup_anndata
    adata_manager.register_fields(adata, **kwargs)
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/_manager.py", line 185, in register_fields
    self._add_field(
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/_manager.py", line 220, in _add_field
    field_registry[_constants._STATE_REGISTRY_KEY] = field.transfer_field(
                                                     ^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_dataframe_field.py", line 221, in transfer_field
    raise ValueError(
ValueError: Category query not found in source registry. Cannot transfer setup without `extend_categories = True`.
