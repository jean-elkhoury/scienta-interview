# import anndata as ad
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sknetwork.clustering import Louvain
from tqdm import tqdm
from sklearn.metrics import (
    adjusted_rand_score as ari,
    normalized_mutual_info_score as nmi,
)
from scienta.models import inVAE
from ray import tune
import mlflow


class Trainer:
    def __init__(self, model: inVAE, beta: float, beta_tc: float, **optimizer_kwargs):
        self.optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        self.model = model
        self.beta = beta
        self.beta_tc = beta_tc
        self.gradient_steps = 0

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        is_warmup: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
        """"""
        beta = self.beta if not is_warmup else 0.0
        beta_tc = self.beta_tc if not is_warmup else 0.0
        counts, b_cov_data, t_cov_data = batch
        outputs = self.model.forward(x=counts, b_cov=b_cov_data, t_cov=t_cov_data)
        batch_loss = self.model.loss(
            x=counts, outputs=outputs, beta=beta, beta_tc=beta_tc
        )

        # Compute metrics for this batch
        batch_metrics = self.compute_metrics(
            outputs=outputs,
            b_cov_data=b_cov_data,
            t_cov_data=t_cov_data,
        )

        total_loss = batch_loss["loss"]
        total_loss.backward()
        self.optimizer.step()
        self.gradient_steps += 1
        return outputs, batch_loss, batch_metrics

    def run_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        is_training: bool,
        is_warmup: bool = False,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Base method for running an epoch with common logic for training and validation."""
        # Set model mode
        if is_training:
            self.model.train()
            prefix = "train"
        else:
            self.model.eval()
            prefix = "val"

        epoch_loss = {}
        epoch_metrics = {}
        num_batches = 0

        for batch in tqdm(data_loader, desc=f"Looping on {prefix} loader"):
            if is_training:
                # Training: use step method which handles optimization
                outputs, batch_loss_tensor, batch_metrics = self.training_step(
                    batch=batch, is_warmup=is_warmup
                )
            else:
                # Validation: forward pass only
                counts, b_cov_data, t_cov_data = batch
                outputs = self.model.forward(
                    x=counts, b_cov=b_cov_data, t_cov=t_cov_data
                )
                beta = self.beta if not is_warmup else 0.0
                beta_tc = self.beta_tc if not is_warmup else 0.0
                batch_loss_tensor = self.model.loss(
                    x=counts, outputs=outputs, beta=beta, beta_tc=beta_tc
                )
                batch_metrics = self.compute_metrics(
                    outputs=outputs,
                    b_cov_data=b_cov_data,
                    t_cov_data=t_cov_data,
                )
            batch_loss = self._convert_tensors_to_scalars(batch_loss_tensor)

            # Accumulate losses and metricsx
            for loss_name, loss_value in batch_loss.items():
                prefixed_loss_name = f"{prefix}_{loss_name}"
                value = epoch_loss.get(prefixed_loss_name, 0.0)
                value += loss_value
                epoch_loss[prefixed_loss_name] = value

            for metric_name, metric_value in batch_metrics.items():
                prefixed_metric_name = f"{prefix}_{metric_name}"
                value = epoch_metrics.get(prefixed_metric_name, 0.0)
                value += metric_value
                epoch_metrics[prefixed_metric_name] = value

            num_batches += 1

        # Compute and log averages
        if num_batches > 0:
            avg_epoch_metrics = {
                metric_name: metric_sum / num_batches
                for metric_name, metric_sum in epoch_metrics.items()
            }
            avg_epoch_loss = {
                loss_name: loss_sum / num_batches
                for loss_name, loss_sum in epoch_loss.items()
            }

            return avg_epoch_loss, avg_epoch_metrics
        else:
            return {}, {}

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        num_epochs_warmup: int,
        early_stopping_patience: int | None = None,
        is_tuning: bool = False,
    ):
        if early_stopping_patience is not None:
            best_val_loss = float("inf")
            patience_counter = 0

        for epoch in range(num_epochs):
            # Training epoch
            train_loss, train_metrics = self.run_epoch(
                train_loader, epoch=epoch, is_training=True
            )

            # Validation epoch
            val_loss, val_metrics = self.run_epoch(
                val_loader, epoch=epoch, is_training=False
            )

            if is_tuning:
                # Give everything to Ray and let it handle mlflow logs
                tune.report(
                    {"epoch": epoch}
                    | val_loss
                    | val_metrics
                    | train_loss
                    | train_metrics
                )
            else:
                # plain mlflow logging
                mlflow.log_params(self.get_params())
                mlflow.log_params(
                    {
                        "num_epochs": num_epochs,
                        "num_epochs_warmup": num_epochs_warmup,
                    }
                )
                mlflow.log_metrics(metrics=train_loss, step=epoch)
                mlflow.log_metrics(metrics=train_metrics, step=epoch)
                mlflow.log_metrics(metrics=val_loss, step=epoch)
                mlflow.log_metrics(metrics=val_metrics, step=epoch)

            if early_stopping_patience is not None:
                current_val_loss = val_loss.get("val_loss", float("inf"))
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def get_params(self) -> dict[str, float]:
        return {
            "beta": self.beta,
            "beta_tc": self.beta_tc,
            "lr": self.optimizer.param_groups[0]["lr"],
            "n_latent_inv": self.model.n_latent_inv,
            "n_latent_spur": self.model.n_latent_spur,
            "n_hidden": self.model.n_hidden,
            "n_layers": self.model.n_layers,
            "dropout_rate": self.model.dropout_rate,
        }

    def _convert_tensors_to_scalars(self, metrics_dict: dict) -> dict[str, float]:
        """Convert PyTorch tensors to Python scalars."""
        converted = {}
        for key, value in metrics_dict.items():
            if hasattr(value, "item"):  # PyTorch tensor
                converted[key] = value.item()
            else:
                converted[key] = float(value)
        return converted

    def louvain_clusters(self, features: np.ndarray):
        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(features)
        # clust.fit(full_counts)
        graph = knn.kneighbors_graph(features)
        clust = Louvain()
        clust.fit(graph)
        return clust.labels_

    def compute_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        b_cov_data: torch.Tensor,
        t_cov_data: torch.Tensor,
    ) -> dict[str, float]:
        louvain_labels = {}
        ground_truth_labels = {}

        ground_truth_labels["bio"] = b_cov_data.detach().cpu().numpy().argmax(axis=1)
        ground_truth_labels["batch"] = t_cov_data.detach().cpu().numpy().argmax(axis=1)
        metric_dict = {"ari": ari, "nmi": nmi}
        metrics = {}
        for representation in "inv", "spur":
            louvain_labels[representation] = self.louvain_clusters(
                features=outputs[f"q_mean_{representation}"].detach().cpu().numpy()
            )
            metrics[f"n_clust_{representation}"] = louvain_labels[representation].max()
            for label_name, label_values in ground_truth_labels.items():
                for metric, metric_func in metric_dict.items():
                    metrics[f"{metric}_{representation}_{label_name}"] = metric_func(
                        louvain_labels[representation], label_values
                    )
        return metrics
