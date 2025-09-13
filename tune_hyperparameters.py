#!/usr/bin/env python3
"""
Ray Tune hyperparameter optimization script for inVAE model.

This script performs hyperparameter search using Ray Tune to find the best
configuration for the inVAE model on the pancreas dataset.
"""

import os
import torch
import scanpy as sc
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import CLIReporter

from scienta.models import inVAE
from scienta.datasets import AnnDataset
from scienta.trainer import Trainer
from scienta.tune_config import (
    get_search_space,
    get_tune_config,
)


def load_data():
    """Load and prepare the dataset."""
    # Load data
    adata = sc.read(
        "data/pancreas.h5ad",
        backup_url="https://www.dropbox.com/s/qj1jlm9w10wmt0u/pancreas.h5ad?dl=1",
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
    )

    # Create trainer
    trainer = Trainer(model=model, beta=config["beta"], lr=config["lr"])

    # Train with Ray Tune integration
    trainer.train_with_tune(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(config["num_epochs"]),
        config=config,
    )


def main():
    """Main function to run hyperparameter optimization."""
    import mlflow

    # Initialize Ray
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")

    # Get configuration
    tune_config = get_tune_config()
    search_space = get_search_space()

    # Start MLflow parent run for the entire hyperparameter search
    with mlflow.start_run(run_name="invae_hyperparameter_search"):
        mlflow.log_params(
            {
                "num_samples": tune_config["num_samples"],
                "max_concurrent_trials": tune_config["max_concurrent_trials"],
                "time_budget_s": tune_config["time_budget_s"],
                "grace_period": tune_config["grace_period"],
                "reduction_factor": tune_config["reduction_factor"],
            }
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
            storage_path="/Users/jelkhoury/Desktop/perso/scienta/ray_results",
            name="invae_hyperparameter_search",
            resume="AUTO",  # Resume from previous runs if available
        )

    # Print best results
    print("\n" + "=" * 50)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 50)

    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    print(
        f"Best trial final validation ARI (inv-bio): {best_trial.last_result['val_ari_inv_bio']}"
    )
    print(
        f"Best trial final validation ARI (inv-batch): {best_trial.last_result['val_ari_inv_batch']}"
    )

    # Save best configuration
    best_config = best_trial.config
    print("\nBest configuration saved to: ./ray_results/best_config.json")

    # Create a summary of all trials
    results_df = analysis.results_df
    results_df.to_csv("./ray_results/trial_results.csv", index=False)
    print("All trial results saved to: ./ray_results/trial_results.csv")

    return analysis


if __name__ == "__main__":
    # Create results directory
    os.makedirs("./ray_results", exist_ok=True)

    # Run hyperparameter optimization
    analysis = main()
