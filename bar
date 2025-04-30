import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ad = sc.read_h5ad("results/5xFAD_scvi_clustered.h5ad")

# Make sure genotype column exists; if not, set everything to "5xFAD"
# ad.obs["genotype"] = ad.obs.get("genotype", "5xFAD")

# Cross-tabulate counts, convert to fractions within each genotype
ct = (pd.crosstab(ad.obs["cluster"], ad.obs["genotype"])
        .apply(lambda x: x / x.sum(), axis=0)        # column-wise fraction
        .reset_index()
        .melt(id_vars="cluster", var_name="Genotype", value_name="Fraction"))

# Bar plot with seaborn
plt.figure(figsize=(6,4))
sns.barplot(data=ct, x="cluster", y="Fraction", hue="Genotype")
plt.title("Cluster abundance by genotype")
plt.ylabel("Fraction of cells")
plt.xlabel("Leiden cluster")
plt.tight_layout()
plt.savefig("cluster_abundance_by_genotype.png", dpi=300)
plt.show()

print("Saved: cluster_abundance_by_genotype.png")
