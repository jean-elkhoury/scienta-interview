# %%
# import anndata as ad
import scanpy as sc
import torch
import mlflow
from scienta.models import inVAE
from scienta.datasets import AnnDataset
from scienta.trainer import Trainer

# Set up MLflow tracking
mlflow.set_tracking_uri("file:///Users/jelkhoury/Desktop/perso/scienta/mlruns")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
train_val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataset, val_dataset = torch.utils.data.random_split(
    train_val_dataset, [0.8, 0.2]
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=len(val_dataset), shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=len(test_dataset), shuffle=True
)
# %%

n_celltypes = adata.obs["celltype"].nunique()
n_batches = adata.obs["sample"].nunique()

beta = 10.0
lr = 1e-5


def run_experiment(beta: float, lr: float) -> float:
    model = inVAE(
        n_genes=dataset.count.shape[1],
        n_bio_covariates=n_celltypes,
        n_tech_covariates=n_batches,
    ).to(device)

    trainer = Trainer(model=model, lr=lr, beta=beta)
    # Single training mode: will create MLflow run automatically
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        num_epochs_warmup=10,
        early_stopping_patience=10,
        is_tuning=False,  # This triggers MLflow logging
    )


run_experiment(beta=beta, lr=lr)

# %%

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
