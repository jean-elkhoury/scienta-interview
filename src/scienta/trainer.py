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


class Trainer:
    def __init__(self, model: inVAE, beta: float, **optimizer_kwargs):
        self.optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        self.model = model
        self.beta = beta
        self.gradient_steps = 0

    def step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """"""
        counts, b_cov_data, t_cov_data = batch
        outputs = self.model.forward(x=counts, b_cov=b_cov_data, t_cov=t_cov_data)
        loss_dict = self.model.loss(counts, outputs, beta=self.beta)

        # Compute metrics for this batch
        metrics = self.compute_metrics(outputs, b_cov_data, t_cov_data, prefix="train")

        mlflow.log_metrics(loss_dict, step=self.gradient_steps)
        total_loss = loss_dict["loss"]
        total_loss.backward()
        self.optimizer.step()
        self.gradient_steps += 1
        return outputs, metrics

    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.eval()
        val_loss = {
            "val_loss": 0.0,  # TODO debt hardcoded values
            "val_reconstruction_loss": 0.0,
            "val_kl_divergence": 0.0,
        }
        for batch in val_loader:
            counts, b_cov_data, t_cov_data = batch
            outputs = self.model.forward(x=counts, b_cov=b_cov_data, t_cov=t_cov_data)
            loss = self.model.loss(counts, outputs, beta=self.beta)
            for key, value in loss.items():
                val_loss[f"val_{key}"] += value

            metrics = self.compute_metrics(outputs, b_cov_data, t_cov_data)
            mlflow.log_metrics(
                metrics,
                step=self.gradient_steps,
            )
        self.model.train()
        mlflow.log_metrics(val_loss, step=self.gradient_steps)
        return val_loss

    def louvain_clusters(self, features: np.ndarray):
        """"""
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
        prefix: str = "val",
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
            mlflow.log_metric(
                key=f"n_clust_{representation}",
                value=louvain_labels[representation].max(),
                step=self.gradient_steps,
            )
            for label_name, label_values in ground_truth_labels.items():
                for metric, metric_func in metric_dict.items():
                    metrics[f"{prefix}_{metric}_{representation}_{label_name}"] = (
                        metric_func(louvain_labels[representation], label_values)
                    )
        return metrics

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        # num_epochs_warmup: int,
    ):
        with mlflow.start_run():
            for epoch in np.arange(num_epochs):
                # Track training metrics for this epoch
                train_metrics_accumulator = {}
                num_batches = 0

                for batch in tqdm(train_loader):
                    outputs, batch_metrics = self.step(batch)

                    # Training metrics
                    for metric_name, metric_value in batch_metrics.items():
                        if metric_name not in train_metrics_accumulator:
                            train_metrics_accumulator[metric_name] = 0.0
                        train_metrics_accumulator[metric_name] += metric_value
                    num_batches += 1

                # Compute and log average training metrics for this epoch
                if num_batches > 0:
                    avg_train_metrics = {
                        metric_name: metric_sum / num_batches
                        for metric_name, metric_sum in train_metrics_accumulator.items()
                    }
                    mlflow.log_metrics(avg_train_metrics, step=epoch)

                self.evaluate(val_loader)
