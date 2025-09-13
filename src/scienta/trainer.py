import mlflow

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


class Trainer:
    def __init__(self, model: inVAE, beta: float, **optimizer_kwargs):
        self.optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        self.model = model
        self.beta = beta
        self.gradient_steps = 0

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
        """"""
        counts, b_cov_data, t_cov_data = batch
        outputs = self.model.forward(x=counts, b_cov=b_cov_data, t_cov=t_cov_data)
        batch_loss = self.model.loss(x=counts, outputs=outputs, beta=self.beta)

        # Compute metrics for this batch
        batch_metrics = self.compute_metrics(
            outputs,
            b_cov_data,
            t_cov_data,
            prefix="train",
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
                outputs, batch_loss, batch_metrics = self.training_step(batch)
            else:
                # Validation: forward pass only
                counts, b_cov_data, t_cov_data = batch
                outputs = self.model.forward(
                    x=counts, b_cov=b_cov_data, t_cov=t_cov_data
                )
                batch_loss = self.model.loss(x=counts, outputs=outputs, beta=self.beta)
                batch_metrics = self.compute_metrics(
                    outputs,
                    b_cov_data,
                    t_cov_data,
                    prefix=prefix,
                )

            # Accumulate losses and metrics
            for loss_name, loss_value in batch_loss.items():
                if loss_name not in epoch_loss:
                    epoch_loss[loss_name] = 0.0
                epoch_loss[loss_name] += loss_value

            for metric_name, metric_value in batch_metrics.items():
                if metric_name not in epoch_metrics:
                    epoch_metrics[metric_name] = 0.0
                epoch_metrics[metric_name] += metric_value

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
            mlflow.log_metrics(avg_epoch_metrics, step=epoch)
            mlflow.log_metrics(avg_epoch_loss, step=epoch)

            return avg_epoch_loss, avg_epoch_metrics
        else:
            return {}, {}

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        # num_epochs_warmup: int,
    ):
        with mlflow.start_run():
            for epoch in range(num_epochs):
                self.run_epoch(train_loader, epoch=epoch, is_training=True)
                self.run_epoch(val_loader, epoch=epoch, is_training=False)

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
        prefix: str,
    ) -> dict[str, float]:
        louvain_labels = {}
        ground_truth_labels = {}

        ground_truth_labels["bio"] = b_cov_data.detach().numpy().argmax(axis=1)
        ground_truth_labels["batch"] = t_cov_data.detach().numpy().argmax(axis=1)
        metric_dict = {"ari": ari, "nmi": nmi}
        metrics = {}
        for representation in "inv", "spur":
            louvain_labels[representation] = self.louvain_clusters(
                features=outputs[f"q_mean_{representation}"].detach().numpy()
            )
            metrics[f"{prefix}_n_clust_{representation}"] = louvain_labels[
                representation
            ].max()
            for label_name, label_values in ground_truth_labels.items():
                for metric, metric_func in metric_dict.items():
                    metrics[f"{prefix}_{metric}_{representation}_{label_name}"] = (
                        metric_func(louvain_labels[representation], label_values)
                    )
        return metrics

    def train_with_tune(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        config: dict,
    ) -> None:
        """Train the model with Ray Tune integration for hyperparameter optimization."""
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10  # Early stopping patience

        for epoch in range(num_epochs):
            # Training epoch
            train_loss, train_metrics = self.run_epoch(
                train_loader, epoch=epoch, is_training=True
            )

            # Validation epoch
            val_loss, val_metrics = self.run_epoch(
                val_loader, epoch=epoch, is_training=False
            )

            # Report metrics to Ray Tune
            tune.report(
                epoch=epoch,
                train_loss=train_loss.get("loss", 0.0),
                train_reconstruction_loss=train_loss.get("reconstruction_loss", 0.0),
                train_kl_divergence=train_loss.get("kl_divergence", 0.0),
                val_loss=val_loss.get("loss", 0.0),
                val_reconstruction_loss=val_loss.get("reconstruction_loss", 0.0),
                val_kl_divergence=val_loss.get("kl_divergence", 0.0),
                # Add clustering metrics
                train_ari_inv_bio=train_metrics.get("train_ari_inv_bio", 0.0),
                train_ari_inv_batch=train_metrics.get("train_ari_inv_batch", 0.0),
                train_ari_spur_bio=train_metrics.get("train_ari_spur_bio", 0.0),
                train_ari_spur_batch=train_metrics.get("train_ari_spur_batch", 0.0),
                val_ari_inv_bio=val_metrics.get("val_ari_inv_bio", 0.0),
                val_ari_inv_batch=val_metrics.get("val_ari_inv_batch", 0.0),
                val_ari_spur_bio=val_metrics.get("val_ari_spur_bio", 0.0),
                val_ari_spur_batch=val_metrics.get("val_ari_spur_batch", 0.0),
            )

            # Early stopping based on validation loss
            current_val_loss = val_loss.get("loss", float("inf"))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
