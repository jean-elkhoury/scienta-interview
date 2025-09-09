import torch
from sklearn.preprocessing import OneHotEncoder


class AnnDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        adata,
        tech_vars,
        bio_vars,
    ):
        self.adata = adata
        self.count = torch.from_numpy(adata.X)
        self.bio_vars = bio_vars
        self.tech_vars = tech_vars
        self.bio_encoder = OneHotEncoder(sparse_output=False)
        self.bio_encoder.fit(adata.obs[self.bio_vars])
        self.tech_encoder = OneHotEncoder(sparse_output=False)
        self.tech_encoder.fit(adata.obs[self.tech_vars])

    def __getitem__(self, idx):
        tech_encoded = self.tech_encoder.transform(
            self.adata.obs.iloc[[idx]][self.tech_vars]
        ).reshape(-1)
        bio_encoded = self.bio_encoder.transform(
            self.adata.obs.iloc[[idx]][self.bio_vars]
        ).reshape(-1)
        return (
            self.count[idx, :],
            torch.from_numpy(bio_encoded).float(),
            torch.from_numpy(tech_encoded).float(),
        )

    def __len__(self):
        return self.count.shape[0]
