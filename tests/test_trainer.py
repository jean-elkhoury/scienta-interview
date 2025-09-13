"""Tests for the Trainer class."""

import pytest
import torch
from scienta.models import inVAE
from scienta.trainer import Trainer
import numpy as np


class TestTrainer:
    """Test cases for the Trainer class."""

    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        return inVAE(
            n_genes=100,
            n_bio_covariates=5,
            n_tech_covariates=3,
            n_latent_inv=20,
            n_latent_spur=5,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 32
        return (
            torch.randn(batch_size, 100),  # counts
            torch.randn(batch_size, 5),  # bio_cov
            torch.randn(batch_size, 3),  # tech_cov
        )

    def test_trainer_creation(self, sample_model):
        """Test that the trainer can be created."""
        trainer = Trainer(model=sample_model, beta=1.0, beta_tc=1.0, lr=1e-3)

        assert trainer.model == sample_model
        assert trainer.beta == 1.0
        assert trainer.beta_tc == 1.0
        assert trainer.optimizer.param_groups[0]["lr"] == 1e-3
        assert trainer.gradient_steps == 0

    def test_training_step(self, sample_model, sample_batch):
        """Test that training step works."""
        trainer = Trainer(model=sample_model, beta=1.0, beta_tc=1.0, lr=1e-3)

        # Test training step
        outputs, batch_loss, batch_metrics = trainer.training_step(sample_batch)

        # Check outputs
        assert isinstance(outputs, dict)
        assert isinstance(batch_loss, dict)
        assert isinstance(batch_metrics, dict)

        # Check that gradient steps increased
        assert trainer.gradient_steps == 1

    def test_training_step_warmup(self, sample_model, sample_batch):
        """Test training step with warmup (beta=0)."""
        trainer = Trainer(model=sample_model, beta=1.0, beta_tc=1.0, lr=1e-3)

        # Test warmup step
        outputs, batch_loss, batch_metrics = trainer.training_step(
            sample_batch, is_warmup=True
        )

        # Check that KL divergence is 0 during warmup
        assert batch_loss["kl_divergence"] == 0.0

    def test_convert_tensors_to_scalars(self, sample_model):
        """Test the tensor to scalar conversion method."""
        trainer = Trainer(model=sample_model, beta=1.0, beta_tc=1.0, lr=1e-3)

        # Test with mixed tensor and scalar values
        test_dict = {
            "tensor_value": torch.tensor(1.5),
            "scalar_value": 2.3,
            "another_tensor": torch.tensor(0.8),
        }

        converted = trainer._convert_tensors_to_scalars(test_dict)

        # Check that all values are now Python scalars
        for key, value in converted.items():
            assert isinstance(value, (int, float, np.int64, np.float64))
            assert not hasattr(value, "item")  # Not a tensor

        # Check specific values (with tolerance for floating point precision)
        assert abs(converted["tensor_value"] - 1.5) < 1e-6
        assert converted["scalar_value"] == 2.3
        assert abs(converted["another_tensor"] - 0.8) < 1e-6

    def test_get_params(self, sample_model):
        """Test that get_params returns the correct parameters."""
        trainer = Trainer(model=sample_model, beta=2.0, beta_tc=2.0, lr=1e-4)

        params = trainer.get_params()

        expected_keys = [
            "beta",
            "beta_tc",
            "lr",
            "n_latent_inv",
            "n_latent_spur",
            "n_hidden",
            "n_layers",
            "dropout_rate",
        ]

        for key in expected_keys:
            assert key in params

        assert params["beta"] == 2.0
        assert params["beta_tc"] == 2.0
        assert params["lr"] == 1e-4
        assert params["n_latent_inv"] == 20
        assert params["n_latent_spur"] == 5

    def test_compute_metrics(self, sample_model, sample_batch):
        """Test that compute_metrics works."""
        trainer = Trainer(model=sample_model, beta=1.0, beta_tc=1.0, lr=1e-3)

        # Get model outputs
        counts, b_cov_data, t_cov_data = sample_batch
        outputs = sample_model.forward(x=counts, b_cov=b_cov_data, t_cov=t_cov_data)

        # Create proper one-hot encoded data for testing
        batch_size = counts.shape[0]
        # Create structured one-hot data
        b_cov_structured = torch.zeros(batch_size, 5)
        b_cov_structured[
            torch.arange(batch_size), torch.randint(0, 5, (batch_size,))
        ] = 1.0

        t_cov_structured = torch.zeros(batch_size, 3)
        t_cov_structured[
            torch.arange(batch_size), torch.randint(0, 3, (batch_size,))
        ] = 1.0

        # Compute metrics
        metrics = trainer.compute_metrics(outputs, b_cov_structured, t_cov_structured)

        # Check that metrics are computed
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # Check that all values are scalars
        for key, value in metrics.items():
            assert isinstance(value, (int, float, np.int64, np.float64))

    def test_gpu_compatibility(self, sample_batch):
        """Test that trainer works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = inVAE(
            n_genes=100,
            n_bio_covariates=5,
            n_tech_covariates=3,
        ).cuda()

        trainer = Trainer(model=model, beta=1.0, beta_tc=1.0, lr=1e-3)

        # Move batch to GPU
        gpu_batch = tuple(tensor.cuda() for tensor in sample_batch)

        # Test training step on GPU
        outputs, batch_loss, batch_metrics = trainer.training_step(gpu_batch)

        # Check that outputs are on GPU
        for key, value in outputs.items():
            assert value.is_cuda

        # Check that losses are on GPU
        for key, value in batch_loss.items():
            assert value.is_cuda
