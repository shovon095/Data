Traceback (most recent call last):
  File "/home/shouvon/bioinform/heat.py", line 9, in <module>
    sc.pl.rank_genes_groups_heatmap(ad, groupby = "cluster", n_genes= 5, show_gene_labels = True,     # gene names on the y-axis
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/legacy_api_wrap/__init__.py", line 82, in fn_compatible
    return fn(*args_all, **kw)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scanpy/plotting/_tools/__init__.py", line 744, in rank_genes_groups_heatmap
    return _rank_genes_groups_plot(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scanpy/plotting/_tools/__init__.py", line 653, in _rank_genes_groups_plot
    return heatmap(
           ^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/legacy_api_wrap/__init__.py", line 82, in fn_compatible
    return fn(*args_all, **kw)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scanpy/plotting/_anndata.py", line 1189, in heatmap
    categories, obs_tidy = _prepare_dataframe(
                           ^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scanpy/plotting/_anndata.py", line 2078, in _prepare_dataframe
    raise ValueError(
ValueError: groupby has to be a valid observation. Given cluster, is not in observations: ['sample', 'n_genes', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'pct_counts_in_top_50_genes', 'pct_counts_in_top_100_genes', 'pct_counts_in_top_200_genes', 'pct_counts_in_top_500_genes', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'protocol', '_scvi_batch', '_scvi_labels', 'leiden']
