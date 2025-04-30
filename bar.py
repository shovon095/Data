import scanpy as sc, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

ad = sc.read_h5ad("results/5xFAD_scvi_clustered_with_genotype.h5ad")

# cross-tabulate, convert to fractions inside each genotype
ct = (pd.crosstab(ad.obs["leiden"], ad.obs["genotype"])
        .apply(lambda col: col / col.sum(), axis=0)
        .reset_index()
        .melt(id_vars="leiden", var_name="Genotype", value_name="Fraction"))

plt.figure(figsize=(6,4))
sns.barplot(data=ct, x="leiden", y="Fraction", hue="Genotype")
plt.xlabel("Leiden cluster")
plt.ylabel("Fraction of cells")
plt.title("Cluster abundance by genotype")
plt.tight_layout()
plt.savefig("cluster_abundance_by_genotype.png", dpi=300)
plt.show()

print("Saved: cluster_abundance_by_genotype.png")

import scanpy as sc, pandas as pd

ad = sc.read_h5ad("results/5xFAD_scvi_clustered_with_genotype.h5ad")

# pick one interesting cluster, e.g. DAM = "5"
cluster_id = "5"
sub = ad[ad.obs["leiden"] == cluster_id].copy()

sc.tl.rank_genes_groups(
    sub, groupby="genotype", method="wilcoxon", pts=True, key_added="de_FADvsWT"
)

# Grab top 20 FAD-up genes
top_fad = sc.get.rank_genes_groups_df(sub, group="5xFAD", key="de_FADvsWT").head(20)
top_fad.to_csv(f"DE_cluster{cluster_id}_FAD_vs_WT.csv", index=False)
print(f"Wrote DE table: DE_cluster{cluster_id}_FAD_vs_WT.csv")

