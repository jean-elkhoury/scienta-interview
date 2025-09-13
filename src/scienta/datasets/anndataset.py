import torch
from sklearn.preprocessing import OneHotEncoder
import anndata as ad


class AnnDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        adata: ad.AnnData,
        tech_vars: list[str],
        bio_vars: list[str],
    ):
        self.adata = adata
        self.count = torch.from_numpy(adata.X)
        self.bio_vars = bio_vars
        self.tech_vars = tech_vars
        self.bio_encoder = OneHotEncoder(sparse_output=False)
        self.bio_encoder.fit(adata.obs[self.bio_vars])
        self.tech_encoder = OneHotEncoder(sparse_output=False)
        self.tech_encoder.fit(adata.obs[self.tech_vars])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tech_encoded = self.tech_encoder.transform(
            self.adata.obs.iloc[[idx]][self.tech_vars]
        ).reshape(-1)
        bio_encoded = self.bio_encoder.transform(
            self.adata.obs.iloc[[idx]][self.bio_vars]
        ).reshape(-1)

        # Move tensors to device if specified
        count_tensor = self.count[idx, :]
        bio_tensor = torch.from_numpy(bio_encoded).float()
        tech_tensor = torch.from_numpy(tech_encoded).float()

        if self.device is not None:
            count_tensor = count_tensor.to(self.device)
            bio_tensor = bio_tensor.to(self.device)
            tech_tensor = tech_tensor.to(self.device)

        return (count_tensor, bio_tensor, tech_tensor)

    def __len__(self):
        return self.count.shape[0]
