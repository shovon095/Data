#!/usr/bin/env python
"""
Project 5xFAD + WT nuclei into the SCVI reference latent space
WITHOUT adapter fine-tuning, then generate a UMAP.
"""

import scanpy as sc
import scvi
import argparse

def main(args):
    # -----------------------------
    # 1) Load query AnnData
    # -----------------------------
    ad = sc.read_h5ad(args.input)      # data/fad_matched.h5ad

    # -----------------------------
    # 2) Load frozen SCVI reference
    #    (NO adapter layers, NO training)
    # -----------------------------
    model = scvi.model.SCVI.load(args.model_dir, adata=ad)
    ad.obsm["X_scVI_noAdapter"] = model.get_latent_representation()

    # -----------------------------
    # 3) Build kNN graph, Leiden, and UMAP
    # -----------------------------
    sc.pp.neighbors(ad, use_rep="X_scVI_noAdapter")
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution=0.5)  # optional, for colouring

    # -----------------------------
    # 4) Plot UMAP coloured by dataset
    # -----------------------------
    sc.pl.umap(
        ad,
        color=["dataset"],             # or ["dataset", "leiden"]
        title="UMAP (no adapter)",
        save="_no_adapter.png"         # writes to ./figures if Scanpy cfg
    )

    # Save AnnData with the new representation if desired
    ad.write_h5ad(args.output, compression="gzip")
    print(f"[no_adapter] UMAP saved and AnnData written to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True,
                   help="Matched 5xFAD+WT AnnData (H5AD)")
    p.add_argument("--model_dir",  required=True,
                   help="Directory of trained SCVI reference model")
    p.add_argument("--output",     required=True,
                   help="Output .h5ad with X_scVI_noAdapter")
    args = p.parse_args()
    main(args)
