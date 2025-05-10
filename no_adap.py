#!/usr/bin/env python

import scanpy as sc
import pandas as pd
import scvi
import argparse

def main(args):
    # 1) Load matched AnnData
    ad = sc.read_h5ad(args.input)

    # 2) Add protocol column (must match reference model)
    ad.obs["protocol"] = pd.Categorical(
        ["droplet"] * ad.n_obs,  # assume all came from droplet
        categories=["droplet", "facs"]
    )

    # 3) Add dataset column (WT vs FAD) if missing
    if "dataset" not in ad.obs.columns:
        ad.obs["dataset"] = ["FAD" if "FAD" in str(idx) else "WT" for idx in ad.obs_names]

    print("🧬 Dataset distribution:")
    print(ad.obs["dataset"].value_counts())

    # 4) Load pre-trained SCVI reference model (no adapters)
    model = scvi.model.SCVI.load(args.model_dir, adata=ad)

    # 5) Project into SCVI latent space (frozen encoder)
    ad.obsm["X_scVI_noAdapter"] = model.get_latent_representation()

    # 6) Run UMAP and Leiden clustering
    sc.pp.neighbors(ad, use_rep="X_scVI_noAdapter")
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution=0.5)

    # 7) Plot UMAP colored by dataset (WT vs FAD)
    sc.pl.umap(
        ad,
        color=["dataset"],
        title="UMAP without Adapter",
        save="_no_adapter.png"  # saved to ./figures by Scanpy
    )

    # 8) Save projected AnnData
    ad.write_h5ad(args.output, compression="gzip")

    print("✅ UMAP saved to figures/umap_no_adapter.png")
    print("✅ Projected AnnData saved to", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input H5AD (FAD + WT, with protocol)")
    parser.add_argument("--model_dir", required=True, help="SCVI model directory (e.g., scvi_tms)")
    parser.add_argument("--output", required=True, help="Output H5AD path")
    main(parser.parse_args())



