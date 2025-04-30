 python ./analyze_clusters.py   --input   ./5xFAD_scvi_mapped.h5ad   --output  ./5xFAD_scvi_clustered.h5ad
/home/shouvon/bioinform/./analyze_clusters.py:10: FutureWarning: In the future, the default backend for leiden will be igraph instead of leidenalg.

 To achieve the future defaults please pass: flavor="igraph" and n_iterations=2.  directed must also be False to work with igraph's implementation.
  sc.tl.umap(ad); sc.tl.leiden(ad, resolution=0.5)
WARNING: saving figure to file figures/umap_fad_clusters.png
WARNING: saving figure to file figures/rank_genes_groups_leiden_fad_markers.png
Traceback (most recent call last):
  File "/home/shouvon/bioinform/./analyze_clusters.py", line 23, in <module>
    args=p.parse_args(); main(args)
                         ^^^^^^^^^^
  File "/home/shouvon/bioinform/./analyze_clusters.py", line 14, in main
    pd.DataFrame(sc.get.rank_genes_groups_df(ad, group=None)).to_csv(
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/pandas/util/_decorators.py", line 333, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/pandas/core/generic.py", line 3967, in to_csv
    return DataFrameRenderer(formatter).to_csv(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/pandas/io/formats/format.py", line 1014, in to_csv
    csv_formatter.save()
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/pandas/io/formats/csvs.py", line 251, in save
    with get_handle(
         ^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/pandas/io/common.py", line 749, in get_handle
    check_parent_directory(str(handle))
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/pandas/io/common.py", line 616, in check_parent_directory
    raise OSError(rf"Cannot save file into a non-existent directory: '{parent}'")
OSError: Cannot save file into a non-existent directory: 'results'
