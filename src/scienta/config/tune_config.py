"""Ray Tune configuration for hyperparameter search."""

from ray import tune
from typing import Dict, Any


def get_search_space() -> Dict[str, Any]:
    """Define a smaller search space for quick testing."""
    return {
        # "n_latent_inv": tune.choice([20, 30, 40]),
        "n_latent_inv": tune.choice([30]),
        # "n_latent_spur": tune.choice([3, 5, 8]),
        "n_latent_spur": tune.choice([8]),
        # "n_hidden": tune.choice([128, 256]),
        "n_hidden": tune.choice([128]),
        # "n_layers": tune.choice([2, 3]),
        "n_layers": tune.choice([2]),
        "dropout_rate": tune.uniform(0.0, 0.2),
        "beta": tune.uniform(1e0, 1e2),
        "beta_tc": tune.uniform(1e0, 1e2),
        "lr": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([128, 256]),
        "num_epochs": tune.choice([20, 30]),
        "num_epochs_warmup": tune.choice([10, 20]),
    }


def get_tune_config() -> Dict[str, Any]:
    """Get a quick configuration for testing."""
    return {
        "search_space": get_search_space(),
        "num_samples": 2,
        "max_concurrent_trials": 2,
        "time_budget_s": 600,  # 10 minutes
        "grace_period": 3,
        "reduction_factor": 2,
    }


def get_metrics_to_optimize() -> Dict[str, str | list[str]]:
    """Get the metrics to optimize and their modes."""
    return {
        "primary_metric": "val_loss",
        "primary_mode": "min",
        "secondary_metrics": [
            "val_ari_inv_bio",
            "val_ari_spur_batch",
            "val_ari_spur_bio",
            "val_ari_inv_batch",
        ],
        "secondary_modes": ["max", "max", "min", "min"],  # max for bio, min for batch
    }
