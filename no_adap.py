import scanpy as sc
import pandas as pd
import scvi
import argparse

def main(args):
    # 1)  Load query AnnData
    ad = sc.read_h5ad(args.input)

    # 2)  Create a batch column that matches the reference categories
    ad.obs["protocol"] = pd.Categorical(
        ["droplet"] * ad.n_obs,           # constant label
        categories=["droplet", "facs"]    # EXACTLY the two seen in training
    )

    # 3)  Load the frozen SCVI reference (no training, no adapters)
    model = scvi.model.SCVI.load_query_data(
        ad,               # query AnnData
        args.model_dir    # folder that contains model.pt
    )

    # 4)  Obtain the latent embedding
    ad.obsm["X_scVI_noAdapter"] = model.get_latent_representation()

    # 5)  Build kNN graph, UMAP, and (optionally) Leiden clusters
    sc.pp.neighbors(ad, use_rep="X_scVI_noAdapter")
    sc.tl.umap(ad)                     # UMAP coordinates in ad.obsm["X_umap"]
    sc.tl.leiden(ad, resolution=0.5)   # optional colouring

    # 6)  Plot UMAP coloured by dataset (WT vs FAD vs reference)
    sc.pl.umap(
        ad,
        color=["dataset"],             # or ["dataset", "leiden"]
        title="UMAP (no adapter)",
        save="_no_adapter.png"         # Scanpy writes to ./figures/
    )

    # 7)  Save the annotated AnnData (optional)
    ad.write_h5ad(args.output, compression="gzip")
    print("✓ Wrote figures/umap_no_adapter.png and", args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="5xFAD+WT AnnData (H5AD)")
    p.add_argument("--model_dir",  required=True, help="Folder with SCVI reference model")
    p.add_argument("--output",     required=True, help="Output H5AD with X_scVI_noAdapter")
    main(p.parse_args())

