 File scvi_tms/model.pt already downloaded
/home/shouvon/miniconda3/lib/python3.11/site-packages/scvi/data/fields/_base_field.py:63: UserWarning: adata.X does not contain unnormalized count data. Are you sure this is what you want?
  self.validate_field(adata)

/home/shouvon/bioinform/no_adap.py:25: FutureWarning: In the future, the default backend for leiden will be igraph instead of leidenalg.

 To achieve the future defaults please pass: flavor="igraph" and n_iterations=2.  directed must also be False to work with igraph's implementation.
  sc.tl.leiden(ad, resolution=0.5)   # optional colouring

Traceback (most recent call last):
  File "/home/shouvon/bioinform/no_adap.py", line 44, in <module>
    main(p.parse_args())
  File "/home/shouvon/bioinform/no_adap.py", line 28, in main
    sc.pl.umap(
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scanpy/plotting/_tools/scatterplots.py", line 686, in umap
    return embedding(adata, "umap", **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scanpy/plotting/_tools/scatterplots.py", line 279, in embedding
    color_source_vector = _get_color_source_vector(
                          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scanpy/plotting/_tools/scatterplots.py", line 1200, in _get_color_source_vector
    values = adata.obs_vector(value_to_plot, layer=layer)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/anndata/_core/anndata.py", line 1297, in obs_vector
    return get_vector(self, k, "obs", "var", layer=layer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/anndata/_core/index.py", line 245, in get_vector
    raise KeyError(msg)
KeyError: 'Could not find key dataset in .var_names or .obs.columns.'
