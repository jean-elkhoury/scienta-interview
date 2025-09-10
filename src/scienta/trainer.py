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
    def __init__(self, model: inVAE, **optimizer_kwargs):
        self.optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        self.model = model
        self.gradient_steps = 0

    def step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """"""
        counts, b_cov_data, t_cov_data = batch
        outputs = self.model.forward(x=counts, b_cov=b_cov_data, t_cov=t_cov_data)
        loss_dict = self.model.loss(counts, outputs, beta=self.beta_scheduled)

        mlflow.log_metrics(loss_dict, step=self.gradient_steps)
        total_loss = loss_dict["loss"]
        total_loss.backward()
        self.optimizer.step()
        self.gradient_steps += 1
        return outputs

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
            loss = self.model.loss(counts, outputs, self.beta_scheduled)
            for key, value in loss.items():
                val_loss[f"val_{key}"] += value
            louvain_labels = self.louvain_clusters(
                features=outputs["q_mean_inv"].detach().numpy()
            )
            celltypes = b_cov_data.detach().numpy().argmax(axis=1)
            batches = t_cov_data.detach().numpy().argmax(axis=1)
            ari_celltype = ari(louvain_labels, celltypes)
            ari_batches = ari(louvain_labels, batches)
            nmi_celltype = nmi(louvain_labels, celltypes)
            nmi_batches = nmi(louvain_labels, batches)
            mlflow.log_metrics(
                {
                    "val_ari_celltype": ari_celltype,  # TODO debt hardcoded values
                    "val_nmi_celltype": nmi_celltype,
                    "val_ari_batches": ari_batches,
                    "val_nmi_batches": nmi_batches,
                },
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

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        num_epochs_warmup: int,
    ):
        with mlflow.start_run():
            for epoch in np.arange(num_epochs):
                self.beta_scheduled = (
                    max(1.0, float(epoch / num_epochs_warmup)) * self.model.beta
                )
                for batch in tqdm(train_loader):
                    self.step(batch)
                self.evaluate(val_loader)
