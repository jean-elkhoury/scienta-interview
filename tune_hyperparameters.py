#!/usr/bin/env python3
"""
Ray Tune hyperparameter optimization script for inVAE model.

This script performs hyperparameter search using Ray Tune to find the best
configuration for the inVAE model on the pancreas dataset.
"""

# %%
from datetime import datetime

import mlflow
import scanpy as sc
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.mlflow import MLflowLoggerCallback

from scienta.config.tune_config import (
    get_search_space,
    get_tune_config,
)
from scienta.datasets import AnnDataset
from scienta.models import inVAE
from scienta.trainer import Trainer
from scienta.utils import DATA_PATH, MLRUNS_PATH, RAY_RESULTS_PATH


def load_data():
    """Load and prepare the dataset."""

    # Load data
    adata = sc.read(
        DATA_PATH,
    )
    del adata.raw

    # Create dataset
    dataset = AnnDataset(adata, tech_vars=["sample"], bio_vars=["celltype"])

    # Split data
    train_val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [0.8, 0.2]
    )

    # Get dataset info
    n_celltypes = adata.obs["celltype"].nunique()
    n_batches = adata.obs["sample"].nunique()
    n_genes = dataset.count.shape[1]

    return train_dataset, val_dataset, test_dataset, n_genes, n_celltypes, n_batches


def train_invae_tune(config):
    """Training function for Ray Tune."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset, val_dataset, test_dataset, n_genes, n_celltypes, n_batches = (
        load_data()
    )

    # Create data loaders with configurable batch size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False
    )

    # Create model with hyperparameters from config
    model = inVAE(
        n_genes=n_genes,
        n_bio_covariates=n_celltypes,
        n_tech_covariates=n_batches,
        n_latent_inv=int(config["n_latent_inv"]),
        n_latent_spur=int(config["n_latent_spur"]),
        n_hidden=int(config["n_hidden"]),
        n_layers=int(config["n_layers"]),
        dropout_rate=config["dropout_rate"],
    ).to(device)

    # Create trainer
    trainer = Trainer(
        model=model, beta=config["beta"], beta_tc=config["beta_tc"], lr=config["lr"]
    )

    # Train with Ray Tune integration
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(config["num_epochs"]),
        num_epochs_warmup=int(config["num_epochs_warmup"]),
        early_stopping_patience=10,
        is_tuning=True,
    )


# %%
"""Main function to run hyperparameter optimization."""

# Initialize Ray
if not torch.cuda.is_available():
    print("CUDA not available, using CPU")

# Set up MLflow tracking
mlflow.set_tracking_uri(MLRUNS_PATH)

# Get configuration
tune_config = get_tune_config()
search_space = get_search_space()
sweep_name = f"invae_hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create MLflow logger callback for Ray Tune
mlflow_logger = MLflowLoggerCallback(
    tracking_uri=MLRUNS_PATH,
    experiment_name=sweep_name,
    tags={"project": "invae", "optimization": "ray_tune"},
)

# Create scheduler
scheduler = ASHAScheduler(
    metric="val_loss",
    mode="min",
    max_t=tune_config["time_budget_s"],
    grace_period=tune_config["grace_period"],
    reduction_factor=tune_config["reduction_factor"],
)

# Create searcher
searcher = OptunaSearch(metric="val_loss", mode="min")

# Create reporter
reporter = CLIReporter(
    metric_columns=[
        "train_loss",
        "val_loss",
        "val_ari_inv_bio",
        "val_ari_inv_batch",
    ]
)

# Run hyperparameter optimization
analysis = tune.run(
    train_invae_tune,
    config=search_space,
    num_samples=tune_config["num_samples"],
    max_concurrent_trials=tune_config["max_concurrent_trials"],
    time_budget_s=tune_config["time_budget_s"],
    scheduler=scheduler,
    search_alg=searcher,
    progress_reporter=reporter,
    callbacks=[mlflow_logger],  # Add MLflow logger callback
    storage_path=RAY_RESULTS_PATH,
    name=sweep_name,
    resume="AUTO",  # Resume from previous runs if available
)

print(f"Best trial config: {analysis.get_best_config(metric='val_loss', mode='min')}")
print(
    f"Best trial final validation loss: {analysis.get_best_trial().last_result['val_loss']}"
)

# %%
