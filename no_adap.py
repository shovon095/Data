import scanpy as sc
import pandas as pd
import scvi
import argparse

def main(args):
    # 1) Load matched 5xFAD + WT AnnData
    ad = sc.read_h5ad(args.input)

    # 2) Add required protocol column for SCVI (must match reference)
    ad.obs["protocol"] = pd.Categorical(
        ["droplet"] * ad.n_obs,
        categories=["droplet", "facs"]
    )

    # 3) Add dataset column for coloring UMAP (WT vs FAD)
    if "dataset" not in ad.obs.columns:
        # If not already present, try to infer from cell names
        ad.obs["dataset"] = ["FAD" if "FAD" in idx else "WT" for idx in ad.obs_names]

    # 4) Load pre-trained SCVI model (frozen encoder)
    model = scvi.model.SCVI.load(args.model_dir, adata=ad)

    # 5) Extract latent space without adapter fine-tuning
    ad.obsm["X_scVI_noAdapter"] = model.get_latent_representation()

    # 6) UMAP and clustering
    sc.pp.neighbors(ad, use_rep="X_scVI_noAdapter")
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution=0.5)  # optional

    # 7) Plot UMAP colored by dataset
    sc.pl.umap(ad, color=["dataset"], save="_no_adapter.png")

    # 8) Save latent-annotated AnnData
    ad.write_h5ad(args.output, compression="gzip")

    print("✓ Done. UMAP saved to figures/umap_no_adapter.png")
    print("✓ AnnData written to", args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="5xFAD+WT matched H5AD")
    p.add_argument("--model_dir",  required=True, help="Directory with trained SCVI model")
    p.add_argument("--output",     required=True, help="Output H5AD path with latent space")
    main(p.parse_args())


