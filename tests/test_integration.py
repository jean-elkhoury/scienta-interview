"""Integration tests for the complete pipeline."""

import pytest
import torch
import numpy as np
import pandas as pd
import anndata as ad
from scienta.models import inVAE
from scienta.datasets import AnnDataset
from scienta.trainer import Trainer


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for integration testing."""
        n_cells = 200
        n_genes = 100

        # Create random count matrix
        X = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)

        # Create observation data
        obs = pd.DataFrame(
            {
                "celltype": np.random.choice(["A", "B", "C", "D"], n_cells),
                "sample": np.random.choice(["s1", "s2", "s3", "s4"], n_cells),
            }
        )

        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs)
        return adata

    def test_complete_training_pipeline(self, sample_data):
        """Test the complete training pipeline."""
        # Create dataset
        dataset = AnnDataset(
            adata=sample_data, tech_vars=["sample"], bio_vars=["celltype"]
        )

        # Split data
        train_val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [0.8, 0.2]
        )
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_val_dataset, [0.8, 0.2]
        )

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )

        # Create model
        n_celltypes = sample_data.obs["celltype"].nunique()
        n_batches = sample_data.obs["sample"].nunique()

        model = inVAE(
            n_genes=dataset.count.shape[1],
            n_bio_covariates=n_celltypes,
            n_tech_covariates=n_batches,
        )

        # Create trainer
        trainer = Trainer(model=model, beta=1.0, beta_tc=1.0, lr=1e-3)

        # Test a few training steps
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Only test a few batches
                break

            outputs, batch_loss, batch_metrics = trainer.training_step(batch)

            # Check that training step works
            assert isinstance(outputs, dict)
            assert isinstance(batch_loss, dict)
            assert isinstance(batch_metrics, dict)

            # Check that gradient steps increased
            assert trainer.gradient_steps == i + 1

    def test_validation_pipeline(self, sample_data):
        """Test the validation pipeline."""
        # Create dataset
        dataset = AnnDataset(
            adata=sample_data, tech_vars=["sample"], bio_vars=["celltype"]
        )

        # Create data loader
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=False
        )

        # Create model
        n_celltypes = sample_data.obs["celltype"].nunique()
        n_batches = sample_data.obs["sample"].nunique()

        model = inVAE(
            n_genes=dataset.count.shape[1],
            n_bio_covariates=n_celltypes,
            n_tech_covariates=n_batches,
        )

        # Create trainer
        trainer = Trainer(model=model, beta=1.0, beta_tc=1.0, lr=1e-3)

        # Test validation epoch
        val_loss, val_metrics = trainer.run_epoch(
            val_loader, epoch=0, is_training=False
        )

        # Check that validation works
        assert isinstance(val_loss, dict)
        assert isinstance(val_metrics, dict)
        assert len(val_loss) > 0
        assert len(val_metrics) > 0

    def test_gpu_pipeline(self, sample_data):
        """Test the complete pipeline on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create dataset (will automatically use GPU)
        dataset = AnnDataset(
            adata=sample_data, tech_vars=["sample"], bio_vars=["celltype"]
        )

        # Create data loader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Create model on GPU
        n_celltypes = sample_data.obs["celltype"].nunique()
        n_batches = sample_data.obs["sample"].nunique()

        model = inVAE(
            n_genes=dataset.count.shape[1],
            n_bio_covariates=n_celltypes,
            n_tech_covariates=n_batches,
        ).cuda()

        # Create trainer
        trainer = Trainer(model=model, beta=1.0, beta_tc=1.0, lr=1e-3)

        # Test training step on GPU
        batch = next(iter(train_loader))
        outputs, batch_loss, batch_metrics = trainer.training_step(batch)

        # Check that everything is on GPU
        for key, value in outputs.items():
            assert value.is_cuda

        for key, value in batch_loss.items():
            assert value.is_cuda

    def test_metrics_computation(self, sample_data):
        """Test that metrics are computed correctly."""
        # Create dataset
        dataset = AnnDataset(
            adata=sample_data, tech_vars=["sample"], bio_vars=["celltype"]
        )

        # Create model
        n_celltypes = sample_data.obs["celltype"].nunique()
        n_batches = sample_data.obs["sample"].nunique()

        model = inVAE(
            n_genes=dataset.count.shape[1],
            n_bio_covariates=n_celltypes,
            n_tech_covariates=n_batches,
        )

        # Create trainer
        trainer = Trainer(model=model, beta=1.0, beta_tc=1.0, lr=1e-3)

        # Get a batch
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=32)))
        counts, b_cov_data, t_cov_data = batch

        # Get model outputs
        outputs = model.forward(x=counts, b_cov=b_cov_data, t_cov=t_cov_data)

        # Compute metrics
        metrics = trainer.compute_metrics(outputs, b_cov_data, t_cov_data)

        # Check that clustering metrics are computed
        expected_metrics = [
            "n_clust_inv",
            "n_clust_spur",
            "ari_inv_bio",
            "ari_inv_batch",
            "nmi_inv_bio",
            "nmi_inv_batch",
            "ari_spur_bio",
            "ari_spur_batch",
            "nmi_spur_bio",
            "nmi_spur_batch",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, np.int64, np.float64))
