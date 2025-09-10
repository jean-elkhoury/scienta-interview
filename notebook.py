# %%
# import anndata as ad
import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import NearestNeighbors
from sknetwork.clustering import Louvain
from scienta.models import inVAE
from scienta.datasets import AnnDataset
from scienta.trainer import Trainer

# 1. Load and explore
# %%
adata = sc.read(
    "data/pancreas.h5ad",
    backup_url="https://www.dropbox.com/s/qj1jlm9w10wmt0u/pancreas.h5ad?dl=1",
)
del adata.raw
# Contains AnnData object (https://anndata.readthedocs.io/en/stable/)

# %%
print(f"Shape: {adata.shape}")
print(f"Cell types: {adata.obs.celltype.value_counts()}")
print(f"Batches: {adata.obs.batch.value_counts()}")
# %%

dataset = AnnDataset(adata, tech_vars=["sample"], bio_vars=["celltype"])
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=len(val_dataset), shuffle=True
)
# %%

n_celltypes = adata.obs["celltype"].nunique()
n_batches = adata.obs["sample"].nunique()
beta = 1.0
lr = 1e-5

model = inVAE(
    n_input=adata.shape[1],
    n_bio_covariates=n_celltypes,
    n_tech_covariates=n_batches,
    beta=beta,
)
trainer = Trainer(model=model, lr=lr)

# %%
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    num_epochs_warmup=5,
)


# %%
def louvain_clusters(features: np.ndarray):
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(features)
    # clust.fit(full_counts)
    graph = knn.kneighbors_graph(features)
    clust = Louvain()
    clust.fit(graph)
    return clust.labels_


# ari_result = ari(cluster, celltype)


ari(adata.obs["sample"], adata.obs["louvain"])

# %%
ari(
    adata.obs["sample"],
    np.random.randint(low=0, high=20, size=adata.obs["sample"].shape),
)


# 2. Standard preprocessing of data
# sc.pp.filter_genes(adata, min_cells=10)
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.highly_variable_genes(adata)
# adata = adata[:, adata.var["highly_variable"]]
# %%
# sc.pp.normalize_total(adata)
# adata.X = np.nan_to_num(adata.X)

# sc.pp.log1p(adata)
# %%
# sc.pl.umap(adata, color="celltype")
# sc.pl.umap(adata, color="sample")
