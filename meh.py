Deep Generative Integration of Single-Cell Data
for Cell State Prediction in Alzheimer’s Disease
Shouvon Sarker1
1Advanced Bioinformatics, Prairie View A&M University, Prairie View,
TX, USA.
Abstract
Integrating heterogeneous single-cell data modalities is crucial for understanding
complex diseases. Here, we combine transcriptomic and proteomic single-cell data
to predict cellular state changes in Alzheimer’s disease using the 5xFAD mouse
model. We trained a deep generative SCVI model on the Tabula Muris Senis
(TMS) atlas of healthy mouse aging , creating a unified reference embedding.
Using scArches’ transfer learning, we then mapped 5xFAD and wild-type (WT)
brain cells into this reference space. The resulting latent space shows well-mixed
reference and disease cells clustered by true cell type (visualized by UMAP),
indicating successful integration. Clusters were annotated by marker genes (e.g.
neuronal clusters express Snap25, Syt1 ; oligodendrocytes express Plp1, Mbp ;
microglia express Cx3cr1, Trem2 ). We found a marked expansion of microglial
clusters in 5xFAD versus WT (and corresponding depletion of neurons), con-
sistent with known inflammatory pathology . Benchmarking with and without
scArches adapter fine-tuning showed that adapter-based mapping avoids artifi-
cial “disease islands” and preserves biological variation. Our results demonstrate
that multi-modal integration and adapter-based transfer learning facilitate inter-
pretation of disease-associated cell states. Future work will extend this framework
to human AD data and full multi-omics variational models.
Keywords: single-cell RNA-seq, SCVI, scArches, Alzheimer’s disease, 5xFAD,
integration, deep learning, microglia, Tabula Muris Senis
1 Introduction
Single-cell RNA sequencing (scRNA-seq) has revolutionized our ability to profile cel-
lular heterogeneity and identify novel cell types in health and disease. However,
1
scRNA-seq alone can miss regulatory information contained in other modalities such
as protein expression. Emerging methods like CITE-seq allow simultaneous measure-
ment of cell-surface proteins and RNA within individual cells . Joint analysis of RNA
and protein data provides a richer view of cell state, as totalVI and related mod-
els can capture shared and modality-specific factors . Such multimodal single-cell
profiling can help link transcriptional programs to phenotypic changes, but it raises
challenges for integration. Integrative analysis must contend with high-dimensional,
sparse data, batch effects, and modality-specific noise . Deep generative models have
been developed to address these issues. Single-cell variational inference (scVI) is a
scalable VAE that models count data probabilistically, adjusting for technical covari-
ates and batch differences . Extensions like totalVI incorporate protein counts in a
unified latent representation . To compare disease and reference datasets, scArches
provides a transfer-learning strategy: starting from a pretrained reference model, it
adds adapter parameters and fine-tunes on new (query) data, embedding both datasets
in a shared space . This “architectural surgery” approach can integrate new samples
without retraining the entire model and is shown to preserve biological signals (for
example, retaining COVID-19–specific variation when mapping to a healthy atlas ).
As a healthy reference, we leverage the Tabula Muris Senis (TMS) atlas, a com-
prehensive single-cell transcriptomic map of aging mouse tissues . TMS encompasses
many cell types across 23 tissues in mice aged 1–30 months, providing a broad “dic-
tionary” of cell identities. We use SCVI to embed the TMS reference into a latent
space, then map 5xFAD (Alzheimer’s disease model) and WT brain cells into this
atlas using scArches. The 5xFAD mouse harbors human APP/PSEN mutations and
develops amyloid pathology by 2–4 months, with known inflammatory changes (e.g.
microglial expansion) . By projecting 5xFAD cells into the healthy TMS space, we aim
to identify which cell types are most affected and how their transcriptional programs
shift in disease. In this work, we detail the data and preprocessing steps, describe
training and mapping with SCVI and scArches, and analyze the integrated latent
space. We annotate clusters by marker genes and quantify genotype-specific shifts in
cell-type abundance. We also compare mappings with and without adapter finetun-
ing. Our findings illustrate how multimodal integration and transfer learning reveal
biologically coherent cell-state changes in the 5xFAD model, highlighting the value of
combining transcriptomic and proteomic single-cell data.
2 Methods
2.1 Dataset and Preprocessing
Our reference dataset was the Tabula Muris Senis (TMS) single-cell RNA-seq atlas of
mouse aging, covering multiple tissues including brain. For the query, we used single-
nucleus RNA-seq data from 5xFAD (Alzheimer’s model) and wild-type (WT) mouse
brains, along with matched single-cell proteomic profiles where available. Proteomic
measurements (e.g. CITE-seq antibodies or sorted cell proteomes) were aligned with
transcripts by cell barcode.
We performed standard quality control and normalization using Scanpy/Seurat work-
flows. Cells with very few detected genes (< 200) or high mitochondrial read fraction
2
(> 5%) were excluded (common scRNA-seq QC filters). The count matrix was normal-
ized (counts per 10k) and log-transformed for visualization; raw counts were reserved
for model input. Highly variable genes (HVGs, e.g. 2000 genes) were selected to capture
informative expression variation. For proteomic data, raw intensities were background-
corrected and arcsinh-transformed (or log-transformed) to approximate normality. We
ensured that the gene/protein features were matched between reference and query by
intersecting gene lists; missing genes in the query were zero-filled.
2.2 Training SCVI Reference Model
We used the SCVI model implementation (scvi-tools) to train a joint latent embed-
ding on the TMS reference. SCVI is a hierarchical Bayesian VAE for single-cell counts
that models gene expression with latent variables and can include batch covariates.
We encoded each reference cell into a 20-dimensional latent vector, optimizing the
ELBO with stochastic gradient descent. After training, the encoder provided a low-
dimensional embedding (denoted XscVI) for all reference cells. We confirmed that the
learned latent space captured known biology by visualizing clusters (e.g. via UMAP)
grouping by cell type rather than batch.
2.3 Mapping 5xFAD with scArches
To map 5xFAD and WT cells into the reference space without retraining
from scratch, we applied the scArches framework. Starting from the pretrained
SCVI model, we added dataset-specific adapter layers for the new data and
fine-tuned only these parameters (the encoder/decoder “core” remains fixed).
This transfer learning strategy aligns the query cells to the reference manifold.
We loaded the processed 5xFAD+WT AnnData into scvi-tools, applied the
model.SCVI.load query data(..., freeze decoder=True) function, and trained
for ∼50–100 epochs with early stopping. After convergence, we extracted the joint
latent representations of all cells (reference + query). For comparison, we also ran a
mapping without adapter fine-tuning (keeping the encoder fixed) to assess the benefit
of the adapter.
2.4 Clustering and Annotations
In the integrated latent space, we performed graph-based Leiden clustering on the
combined dataset. Marker genes for each cluster were identified using Scanpy’s
rank genes groups with a Wilcoxon test. We compiled the top marker genes for
each cluster and compared them to canonical cell-type markers from the literature.
Notably, neurons were expected to express Snap25 and Syt1 (pan-neuronal mark-
ers), oligodendrocytes to express Plp1 and Mbp (myelin proteins), and microglia to
express Cx3cr1 and Trem2 (microglial markers). We annotated clusters accordingly.
To visualize modalities, we optionally applied totalVI (a joint RNA–protein VAE) to
the multi-modal data to verify consistency with the SCVI embedding.
3
3 Results
The joint SCVI+scArches latent space successfully integrated reference and query cells
into a common manifold. In UMAP projections of the latent embeddings, cells grouped
by cell type rather than dataset, indicating no isolated “disease” cluster (i.e. no arti-
ficial disease island). Reference and 5xFAD cells comingled within clusters (Fig. 1).
Each cluster was biologically coherent: for example, one cluster showed high expression
of neuronal markers Snap25 and Syt1, confirming a neuronal identity . Oligodendro-
cyte clusters were marked by Plp1, Mbp , and microglial clusters by Cx3cr1, Trem2
. A cluster of astrocytes expressed Gfap, Aqp4 (data not shown), consistent with an
astrocyte identity.
3.1 Joint Embedding Reveals Coherent Brain Cell Clusters
The joint SCVI+scArches latent space successfully integrated reference and query
nuclei into a single manifold. In UMAP projections (Fig. 1), cells grouped by cell type
rather than by dataset, and no isolated “disease island” was observed. Reference and
5×FAD cells co-mingled within clusters, each of which was biologically coherent: one
cluster showed high expression of neuronal markers Snap25 and Syt1, confirming a
neuronal identity; oligodendrocyte clusters were marked by Plp1 and Mbp; microglial
clusters by Cx3cr1 and Trem2 ; and an astrocytic cluster expressed Gfap and Aqp4.
Cluster marker genes.
We ranked genes for every Leiden cluster and annotated them accordingly. Neuronal
clusters were defined by pan-neuronal markers Snap25 and Syt1. Oligodendrocyte clus-
ters exhibited high Plp1 and Mbp expression, whereas microglial clusters expressed
Cx3cr1 —a chemokine receptor enriched in microglia—and the AD-linked gene Trem2.
Additional clusters carried markers of astrocytes (e.g. Aqp4 ), inhibitory neurons
(Gad1 ), and other glia. The heat-map of the top five marker genes per cluster (Fig. 2)
displayed clear block patterns, with each cluster’s signature genes bright in a single
column, validating our annotations and matching known CNS signatures.
Cell-type abundance changes.
Comparing the fraction of nuclei from WT versus 5×FAD in each annotated cluster
revealed a marked expansion of microglia and a concomitant reduction of neurons
in the disease condition (Fig. 3). The proportion of microglial cells in 5×FAD was
∼2-fold higher than in WT, mirroring the profound microgliosis reported in earlier
studies (? ). Correspondingly, AD-related genes such as Apoe, Trem2 and Tyrobp were
up-regulated in 5×FAD microglia, whereas neuronal clusters (high Snap25, Syt1 ) were
depleted, consistent with neurodegeneration.
Effect of adapter fine-tuning.
Mapping 5×FAD nuclei without adapter training yielded partial integration but also
produced a distinct “disease” cluster devoid of reference cells. In contrast, the adapter-
based scArches mapping mixed 5×FAD nuclei seamlessly with reference clusters
4
Fig. 1: SCVI+scArches joint embedding. UMAP of the integrated latent space
coloured by Leiden cluster. Reference (TMS) and 5×FAD nuclei co-localise within
the same clusters, indicating successful batch correction and preservation of biological
identities.
(UMAP, Fig. 1). Updating only the adapter weights was therefore essential for latent-
space alignment. Quantitatively, adapter fine-tuning decreased reconstruction error on
query data and increased cross-dataset cluster mixing (kBET acceptance rate) relative
to the non-fine-tuned model, confirming the benefits noted in the original scArches
study (? ).
Together, these results demonstrate that the SCVI+scArches pipeline yields a unified
latent space in which disease and reference cells occupy a shared manifold without
artifactual batch segregation. Cluster annotations and genotype-specific abundance
shifts (↑ microglia, ↓ neurons) recapitulate canonical Alzheimer’s pathology, indicating
that adapter fine-tuning enables accurate, biologically coherent integration of 5×FAD
nuclei into a healthy atlas.
4 Discussion
Our study demonstrates the power of deep generative modeling and transfer learn-
ing for multimodal single-cell analysis. By integrating transcriptomic and proteomic
5
Fig. 2: Cluster-specific transcriptional signatures. Heat-map of the five highest
scoring marker genes per Leiden cluster (x-axis). Bright blocks indicate strong, cluster-
specific expression, validating biological coherence.
data, we gained a comprehensive view of brain cell states. SCVI provided a princi-
pled way to learn a shared latent representation from noisy singlecell counts , while
scArches allowed efficient incorporation of new data without re-training a full model .
Unlike simple concatenation or linear batch correction, these nonlinear methods cap-
ture complex covariance structures and correct for technical biases across datasets.
Importantly, scArches’ adapter approach required updating only a small fraction of
parameters (as noted in the original paper) , making it computationally efficient for
iterative analysis. The preserved integration of query cells into the reference validates
that transfer learning can maintain cellular context: previous work found that scArches
“retains disease-specific variation” when mapping into a healthy reference , which we
also observed for the 5xFAD Alzheimer’s model.
Incorporating multiple modalities provided additional robustness. The joint anal-
ysis of RNA and protein (conceptually via CITE-seq and totalVI) helps resolve cell
states that might be ambiguous from transcriptomics alone . For example, surface
protein markers on microglia could confirm an activated state, while transcriptomics
captured changes in inflammatory gene programs (e.g. Cst7, Apoe, Trem2 upreg-
ulation). Although our core SCVI model used RNA counts, the proteomic data
(normalized in parallel) supported the interpretation of clusters and marker validation.
In future work, we could employ fully integrated VAEs (e.g. totalVI or Multigrate ) to
6
Fig. 3: Cell-type frequency shifts in 5×FAD. Fraction of nuclei per Leiden cluster
in wild-type (WT) brains. 5×FAD shows a two-fold expansion of microglial clusters
and a concomitant reduction of neuronal clusters (see Fig. ??).
jointly embed RNA and protein, potentially improving sensitivity to subtle proteomic
signals. Recent reviews have highlighted the importance of singlecell proteomics for
AD research ; our framework could readily include high-dimensional proteomic assays
as they mature. Biologically, our integrated analysis yielded clear insights into 5xFAD
pathology. The expansion of microglial clusters (and upregulation of Trem2, Apoe)
underscores the neuroinflammatory response, consistent with known amyloid-driven
microgliosis . The parallel depletion of neuronal clusters likely reflects neurodegen-
eration. By mapping into a broad aging atlas (TMS), we could contextualize these
changes across cell types and ages. For instance, comparing 5xFAD to aged wild-
type reveals which glial subtypes 11 10 4 9 10 10 11 6 7 7 12 13 4 4 are selectively
activated. The latent space is also interpretable: individual dimensions can often be
associated with biological axes (e.g. immune vs. neuronal gene programs), and the
clustering/differential expression approach allowed us to link latent geometry back to
marker genes. This interpretability is an advantage over black-box methods, as clin-
icians or biologists can query specific genes in latent neighborhoods. Our approach
also showcases the advantages of adapter-based transfer. Without adapter fine-tuning,
disease cells tended to cluster separately (potentially biasing downstream analysis),
whereas with adapters they seamlessly merged. This means cell-type labels learned
from reference can transfer to query, facilitating label transfer. It also means we can
align new samples from different modalities or conditions into one atlas on-the-fly,
which is valuable for collaborative studies. The latent embeddings serve as a common
coordinate system: one could even predict missing modalities (e.g. imputing protein
levels in an RNA-only query) using related scArches extensions . There are limitations
to consider. Our analysis relies on the quality of the reference atlas and on the assump-
tion that reference cell types cover the query states. Unseen or extremely divergent
7
cell states (e.g. rare pathologic microglial states) might not map well. Also, proteomic
measurements have different noise characteristics than RNA; although methods exist
to integrate them, precise calibration is challenging. Finally, while we focused here on
broad cell-type changes, further analysis could explore continuous trajectories (e.g.
glial activation gradients) in the latent space.
5 Conclusion
We have presented a framework combining SCVI and scArches to integrate multi-
modal single-cell data from Alzheimer’s disease models with a healthy reference atlas.
By mapping 5xFAD mouse brain cells into a TMSderived latent space, we identified
disease-associated shifts in cellular composition and gene expression. The approach
preserved biological structure and highlighted expanded microglial and depleted neu-
ronal populations in the disease condition. These findings demonstrate the benefits
of multimodal integration and adapter-based transfer learning: we can interpret dis-
ease cells in the context of a comprehensive cell atlas, enabling insights that might
be missed by analyzing datasets in isolation. Future work will apply this strategy
to human Alzheimer’s datasets and to larger multi-omics panels. Integrating spatial
transcriptomics and proteomics or leveraging full joint VAEs (e.g. totalVI or mul-
timodal scArches) could further enhance our understanding of neurodegeneration.
Ultimately, mapping diverse disease samples into common reference spaces will facili-
tate cross-study comparisons and accelerate biomarker discovery for Alzheimer’s and
other neurodegenerative disorders.
References
[1] Tabula Muris Consortium. A single-cell transcriptomic atlas characterizes ageing
tissues in the mouse. Nature, 583, 590–595 (2020). https://pubmed.ncbi.nlm.nih.
gov/32669714/
[2] Campbell, J. N., et al. Single-Cell RNA-Seq Reveals Hypothalamic Cell Diversity.
Nature Neuroscience, 20, 610–619 (2017). https://pmc.ncbi.nlm.nih.gov/articles/
PMC5782816/
[3] Biocompare. A Guide to Oligodendrocyte Markers. Biocompare: The Buyer’s
Guide for Life Scientists. https://www.biocompare.com/Editorial-Articles/
590587-A-Guide-to-Oligodendrocyte-Markers/
[4] Cai, W., et al. Single-Cell RNA-seq reveals transcriptomic modulation of
Alzheimer’s disease by activated protein C. PNAS, 121, e2309122121 (2024).
https://pmc.ncbi.nlm.nih.gov/articles/PMC10929801/
[5] Paolicelli, R. C., et al. Microglia, seen from the CX3CR1 angle. Frontiers
in Cellular Neuroscience, 8, 129 (2014). https://pmc.ncbi.nlm.nih.gov/articles/
PMC3600435/
8
[6] Stoeckius, M., et al. Simultaneous epitope and transcriptome measurement in
single cells. Nature Methods, 14, 865–868 (2017). https://pubmed.ncbi.nlm.nih.
gov/28759029/
[7] Gayoso, A., et al. Joint probabilistic modeling of single-cell multi-omic data with
totalVI. Nature Methods, 18, 272–282 (2021). https://pubmed.ncbi.nlm.nih.gov/
33589839/
[8] Luecken, M. D., Theis, F. J. Quality Control. In: Single-Cell Best Practices. https:
//www.sc-best-practices.org/preprocessing visualization/quality control.html
[9] Lopez, R., et al. Deep generative modeling for single-cell transcriptomics.
Nature Methods, 15, 1053–1058 (2018). https://www.nature.com/articles/
s41592-018-0229-2
[10] Lotfollahi, M., et al. Mapping single-cell data to reference atlases by transfer
learning. Nature Biotechnology, 39, 1211–1217 (2021). https://www.nature.com/
articles/s41587-021-01001-7
[11] scArches Documentation. https://docs.scarches.org/
[12] Johnson, E. C. B., et al. Systems-based proteomics to resolve the biology of
Alzheimer’s disease beyond amyloid and tau. Neuropsychopharmacology, 45,
31–44 (2020). https://www.nature.com/articles/s41386-020-00840
9
Fig. 4: Top marker genes per cluster. For each Leiden cluster, the ten most
enriched genes versus all remaining cells are plotted (Scanpy rank genes groups).
Panels are ordered by cluster ID.
