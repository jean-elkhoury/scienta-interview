# %%
# import anndata as ad
import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score as ari
from scienta.models import inVAE
from scienta.datasets import AnnDataset
from tqdm import tqdm
import mlflow 
mlflow.autolog()
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

# %%
n_celltypes = adata.obs["celltype"].nunique()
n_batches = adata.obs["sample"].nunique()
model = inVAE(
    n_input=adata.shape[1], n_bio_covariates=n_celltypes, n_tech_covariates=n_batches
)


# %%
# def preprocess(adata):
#     # adata.X = np.floor(adata.X - adata.X.min())  # make counts positive int
#     # n_bio_covariates = adata.obs["celltype"].nunique()
#     # n_tech_covariates = adata.obs["sample"].nunique()
#     # n_cells = adata.shape[0]
#     # b_cov_indices = torch.randint(0, n_bio_covariates, (n_cells,))
#     b_cov_indices = torch.from_numpy(pd.factorize(adata.obs["celltype"])[0])
#     b_cov_data = F.one_hot(b_cov_indices, num_classes=n_bio_covariates).float()

#     # Technical covariates (one-hot encoded)
#     # t_cov_indices = torch.randint(0, n_tech_covariates, (n_cells,))
#     t_cov_indices = torch.from_numpy(pd.factorize(adata.obs["sample"])[0])
#     t_cov_data = F.one_hot(t_cov_indices, num_classes=n_tech_covariates).float()
#     counts = torch.from_numpy(adata.X)
#     return counts, b_cov_data, t_cov_data

dataset = AnnDataset(adata, tech_vars=["sample"], bio_vars=["celltype"])
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# %%


class Trainer:
    def train(self):
        for batch in tqdm(train_loader):
            counts, b_cov_data, t_cov_data = batch
            outputs = model.forward(x=counts, b_cov=b_cov_data, t_cov=t_cov_data)
            loss_dict = model.loss(counts, outputs)
            # outputs[]
            total_loss = loss_dict["loss"]
            total_loss.backward()
            optimizer.step()

Trainer().train()
# %%
ari(adata.obs["sample"], adata.obs["louvain"])

# %%
ari(
    adata.obs["sample"],
    np.random.randint(low=0, high=20, size=adata.obs["sample"].shape),
)
