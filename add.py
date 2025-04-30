#!/usr/bin/env python
"""
Add a 'genotype' column to the 5xFAD AnnData
based on the sample names:  contains '5XFAD'  ->  '5xFAD'
                            otherwise         ->  'WT'
"""

import scanpy as sc
import pandas as pd

# --- 1. Load the clustered AnnData ---
adata_path = "results/5xFAD_scvi_clustered.h5ad"
ad = sc.read_h5ad(adata_path)
print(f"Loaded {adata_path}  →  {ad.n_obs:,} cells")

# --- 2. Create genotype column ---
ad.obs["genotype"] = (
    ad.obs["sample"]
      .str.contains("5XFAD", case=False, regex=False)   # True / False
      .map({True: "5xFAD", False: "WT"})                # label
      .astype("category")
)

# --- 3. Quick sanity check ---
print("\nCell counts per genotype:")
print(ad.obs["genotype"].value_counts())

# --- 4. (Optional) save updated AnnData ---
out_path = "results/5xFAD_scvi_clustered_with_genotype.h5ad"
ad.write_h5ad(out_path, compression="gzip")
print(f"\nSaved updated AnnData with genotype column → {out_path}")
