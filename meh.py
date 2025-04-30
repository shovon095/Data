Traceback (most recent call last):
  File "/home/shouvon/bioinform/./analyze_clusters.py", line 23, in <module>
    args=p.parse_args(); main(args)
                         ^^^^^^^^^^
  File "/home/shouvon/bioinform/./analyze_clusters.py", line 10, in main
    sc.tl.umap(ad); sc.tl.leiden(ad, resolution=0.5)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scanpy/tools/_leiden.py", line 128, in leiden
    _utils.ensure_igraph()
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/scanpy/_utils/__init__.py", line 109, in ensure_igraph
    raise ImportError(msg)
ImportError: Please install the igraph package: `conda install -c conda-forge python-igraph` or `pip3 install igraph`.
