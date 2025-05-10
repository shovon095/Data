-rw-rw-r--  1 shouvon shouvon  883929019 Apr 29 20:29 5xFAD_matched.h5ad
-rw-rw-r--  1 shouvon shouvon  494601109 Apr 29 14:31 5xFAD_RNA.h5ad
-rw-rw-r--  1 shouvon shouvon  890296658 Apr 29 20:52 5xFAD_scvi_mapped.h5ad
drwx------ 14 shouvon shouvon       4096 Apr 29 14:18 5xFAD_snRNA
-rw-rw-r--  1 shouvon shouvon        980 Apr 29 23:04 add.py
-rw-rw-r--  1 shouvon shouvon        901 Apr 29 13:52 analyze_clusters.py
-rw-rw-r--  1 shouvon shouvon        748 Apr 29 23:13 bar.py
-rw-rw-r--  1 shouvon shouvon   76091694 Apr 29 11:53 cleaned_facs_data.h5ad
-rw-rw-r--  1 shouvon shouvon      78854 Apr 29 23:14 cluster_abundance_by_genotype.png
-rw-rw-r--  1 shouvon shouvon        962 Apr 29 12:20 concat.py
drwxrwxr-x  3 shouvon shouvon       4096 May  9 20:24 Data
drwxrwxr-x  2 shouvon shouvon       4096 Apr 29 22:47 figures
-rw-rw-r--  1 shouvon shouvon        657 Apr 29 22:46 heat.py
drwx------  2 shouvon shouvon       4096 Apr 29 12:18 inCITE_mouse
-rw-rw-r--  1 shouvon shouvon       1818 Apr 29 20:38 map_fad.py
-rw-rw-r--  1 shouvon shouvon       1852 May  9 20:24 no_adap.py
-rw-rw-r--  1 shouvon shouvon       3311 Apr 29 14:38 preprocess_fad.py
-rw-rw-r--  1 shouvon shouvon        753 Apr 29 10:43 preprocess.py
drwxrwxr-x  2 shouvon shouvon       4096 Apr 29 23:05 results
drwxrwxr-x  2 shouvon shouvon       4096 Apr 29 13:54 scvi_tms
-rw-rw-r--  1 shouvon shouvon 2368023386 Apr 29 12:25 tms_concat.h5ad
-rw-rw-r--  1 shouvon shouvon 2394166017 Apr 29 13:58 TMS_scVI_integrated.h5ad
-rw-rw-r--  1 shouvon shouvon       1388 Apr 29 12:14 train.py
(base) shouvon@dxs4-DGX-Station:~/bioinform$ vim scvi_tms
(base) shouvon@dxs4-DGX-Station:~/bioinform$ (base) shouvon@dxs4-DGX-Station:~/bioinform$ python no_adap.py     --input   ./5xFAD_matched.h5ad     --model_dir  models/scvi_tms (base) shouvon@dxs4-DGX-Station:~/bioinform$
(base) shouvon@dxs4-DGX-Station:~/bioinform$ python no_adap.py     --input   ./5xFAD_matched.h5ad     --model_dir  ./scvi_tms     --output  results/fad_no_adapter.h5ad
/home/shouvon/miniconda3/lib/python3.11/site-packages/docrep/decorators.py:43: SyntaxWarning: 'param_categorical_covariate_keys' is not a valid key!
  doc = func(self, args[0].__doc__, *args[1:], **kwargs)
/home/shouvon/miniconda3/lib/python3.11/site-packages/docrep/decorators.py:43: SyntaxWarning: 'param_continuous_covariate_keys' is not a valid key!
  doc = func(self, args[0].__doc__, *args[1:], **kwargs)
INFO     File ./scvi_tms/model.pt already downloaded
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
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_dataframe_field.py", line 210, in transfer_field
    self.validate_field(adata_target)
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_dataframe_field.py", line 179, in validate_field
    raise KeyError(f"{self._original_attr_key} not found in adata.{self.attr_name}.")
KeyError: 'protocol not found in adata.obs.'
