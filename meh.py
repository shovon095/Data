total 20830756
-rw-rw-r--  1 shouvon shouvon         260 May  9 21:59 01.py
-rw-rw-r--  1 shouvon shouvon         480 May  9 22:05 02.py
-rw-rw-r--  1 shouvon shouvon   883929019 Apr 29 20:29 5xFAD_matched.h5ad
-rw-rw-r--  1 shouvon shouvon 14223422568 May  9 20:44 5xFAD_matched_with_protocol.h5ad
-rw-rw-r--  1 shouvon shouvon   494601109 Apr 29 14:31 5xFAD_RNA.h5ad
-rw-rw-r--  1 shouvon shouvon   890296658 Apr 29 20:52 5xFAD_scvi_mapped.h5ad
drwx------ 14 shouvon shouvon        4096 Apr 29 14:18 5xFAD_snRNA
-rw-rw-r--  1 shouvon shouvon         331 May  9 20:42 add_1.py
-rw-rw-r--  1 shouvon shouvon         980 Apr 29 23:04 add.py
-rw-rw-r--  1 shouvon shouvon         901 Apr 29 13:52 analyze_clusters.py
-rw-rw-r--  1 shouvon shouvon         748 Apr 29 23:13 bar.py
-rw-rw-r--  1 shouvon shouvon    76091694 Apr 29 11:53 cleaned_facs_data.h5ad
-rw-rw-r--  1 shouvon shouvon       78854 Apr 29 23:14 cluster_abundance_by_genotype.png
-rw-rw-r--  1 shouvon shouvon         962 Apr 29 12:20 concat.py
drwxrwxr-x  3 shouvon shouvon        4096 May  9 22:03 Data
drwxrwxr-x  2 shouvon shouvon        4096 May  9 21:48 figures
-rw-rw-r--  1 shouvon shouvon         657 Apr 29 22:46 heat.py
drwx------  2 shouvon shouvon        4096 Apr 29 12:18 inCITE_mouse
-rw-rw-r--  1 shouvon shouvon        1818 Apr 29 20:38 map_fad.py
-rw-rw-r--  1 shouvon shouvon        1830 May  9 22:03 no_adap.py
-rw-rw-r--  1 shouvon shouvon        3311 Apr 29 14:38 preprocess_fad.py
-rw-rw-r--  1 shouvon shouvon         753 Apr 29 10:43 preprocess.py
drwxrwxr-x  2 shouvon shouvon        4096 May  9 21:48 results
drwxrwxr-x  2 shouvon shouvon        4096 May  9 20:30 scvi_tms
-rw-rw-r--  1 shouvon shouvon  2368023386 Apr 29 12:25 tms_concat.h5ad
-rw-rw-r--  1 shouvon shouvon  2394127204 May  9 21:26 TMS_scVI_integrated.h5ad
-rw-rw-r--  1 shouvon shouvon        1387 May  9 21:13 train.py
(base) shouvon@dxs4-DGX-Station:~/bioinform$
(base) shouvon@dxs4-DGX-Station:~/bioinform$ (base) shouvon@dxs4-DGX-Station:~/bioinform$ python 02.py
Traceback (most recent call last):
  File "/home/shouvon/bioinform/02.py", line 4, in <module>
    ad_wt = sc.read_h5ad("wt.h5ad")
            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/anndata/_io/h5ad.py", line 239, in read_h5ad
    with h5py.File(filename, "r") as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/h5py/_hl/files.py", line 564, in __init__
    fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/shouvon/miniconda3/lib/python3.11/site-packages/h5py/_hl/files.py", line 238, in make_fid
    fid = h5f.open(name, flags, fapl=fapl)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5f.pyx", line 102, in h5py.h5f.open
FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'wt.h5ad', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
