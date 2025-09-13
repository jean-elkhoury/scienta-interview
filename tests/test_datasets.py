"""Tests for the AnnDataset class."""

import pytest
import torch
import numpy as np
import pandas as pd
import anndata as ad
from scienta.datasets import AnnDataset


class TestAnnDataset:
    """Test cases for the AnnDataset class."""

    @pytest.fixture
    def sample_adata(self):
        """Create a sample AnnData object for testing."""
        n_cells = 100
        n_genes = 50

        # Create random count matrix
        X = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)

        # Create observation data
        obs = pd.DataFrame(
            {
                "celltype": np.random.choice(["A", "B", "C"], n_cells),
                "sample": np.random.choice(["s1", "s2", "s3"], n_cells),
            }
        )

        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs)
        return adata

    def test_dataset_creation(self, sample_adata):
        """Test that the dataset can be created."""
        dataset = AnnDataset(
            adata=sample_adata, tech_vars=["sample"], bio_vars=["celltype"]
        )

        assert len(dataset) == sample_adata.n_obs
        assert dataset.count.shape == (sample_adata.n_obs, sample_adata.n_vars)

    def test_dataset_getitem(self, sample_adata):
        """Test that __getitem__ returns the correct format."""
        dataset = AnnDataset(
            adata=sample_adata, tech_vars=["sample"], bio_vars=["celltype"]
        )

        # Get a sample
        count, bio, tech = dataset[0]

        # Check types
        assert isinstance(count, torch.Tensor)
        assert isinstance(bio, torch.Tensor)
        assert isinstance(tech, torch.Tensor)

        # Check shapes
        assert count.shape == (sample_adata.n_vars,)
        assert bio.shape == (len(sample_adata.obs["celltype"].unique()),)
        assert tech.shape == (len(sample_adata.obs["sample"].unique()),)

    def test_dataset_device_placement(self, sample_adata):
        """Test that tensors are moved to the correct device."""
        dataset = AnnDataset(
            adata=sample_adata, tech_vars=["sample"], bio_vars=["celltype"]
        )

        count, bio, tech = dataset[0]

        # Check that tensors are on the expected device
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert count.device == expected_device
        assert bio.device == expected_device
        assert tech.device == expected_device

    def test_dataset_encoding(self, sample_adata):
        """Test that one-hot encoding works correctly."""
        dataset = AnnDataset(
            adata=sample_adata, tech_vars=["sample"], bio_vars=["celltype"]
        )

        # Get all samples
        counts, bios, techs = zip(*[dataset[i] for i in range(len(dataset))])

        # Check that bio and tech encodings are one-hot
        bio_tensor = torch.stack(bios)
        tech_tensor = torch.stack(techs)

        # Each row should sum to 1 (one-hot encoding)
        assert torch.allclose(bio_tensor.sum(dim=1), torch.ones(len(dataset)))
        assert torch.allclose(tech_tensor.sum(dim=1), torch.ones(len(dataset)))

        # Values should be 0 or 1
        assert torch.all((bio_tensor == 0) | (bio_tensor == 1))
        assert torch.all((tech_tensor == 0) | (tech_tensor == 1))

    def test_dataset_with_dataloader(self, sample_adata):
        """Test that the dataset works with PyTorch DataLoader."""
        dataset = AnnDataset(
            adata=sample_adata, tech_vars=["sample"], bio_vars=["celltype"]
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        # Get a batch
        batch = next(iter(dataloader))
        counts, bios, techs = batch

        # Check batch shapes
        assert counts.shape[0] == 16  # batch size
        assert counts.shape[1] == sample_adata.n_vars
        assert bios.shape[0] == 16
        assert techs.shape[0] == 16
