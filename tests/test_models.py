"""Tests for the inVAE model."""

import pytest
import torch
from scienta.models import inVAE


class TestInVAE:
    """Test cases for the inVAE model."""

    def test_model_creation(self):
        """Test that the model can be created with default parameters."""
        model = inVAE(
            n_genes=100,
            n_bio_covariates=5,
            n_tech_covariates=3,
        )
        assert isinstance(model, inVAE)
        assert model.n_latent_inv == 30
        assert model.n_latent_spur == 5
        assert model.n_hidden == 128
        assert model.n_layers == 2
        assert model.dropout_rate == 0.1

    def test_model_creation_custom_params(self):
        """Test model creation with custom parameters."""
        model = inVAE(
            n_genes=200,
            n_bio_covariates=10,
            n_tech_covariates=5,
            n_latent_inv=50,
            n_latent_spur=10,
            n_hidden=256,
            n_layers=3,
            dropout_rate=0.2,
        )
        assert model.n_latent_inv == 50
        assert model.n_latent_spur == 10
        assert model.n_hidden == 256
        assert model.n_layers == 3
        assert model.dropout_rate == 0.2

    def test_forward_pass(self):
        """Test that the model can perform a forward pass."""
        model = inVAE(
            n_genes=100,
            n_bio_covariates=5,
            n_tech_covariates=3,
        )

        batch_size = 32
        x = torch.randn(batch_size, 100)
        b_cov = torch.randn(batch_size, 5)
        t_cov = torch.randn(batch_size, 3)

        outputs = model.forward(x=x, b_cov=b_cov, t_cov=t_cov)

        # Check that all expected outputs are present
        expected_keys = [
            "q_mean_inv",
            "q_log_var_inv",
            "q_mean_spur",
            "q_log_var_spur",
            "px_log_mean",
            "px_log_disp",
            "p_mean_inv",
            "p_log_var_inv",
            "p_mean_spur",
            "p_log_var_spur",
        ]
        for key in expected_keys:
            assert key in outputs
            assert outputs[key].shape[0] == batch_size

    def test_loss_computation(self):
        """Test that the model can compute loss."""
        model = inVAE(
            n_genes=100,
            n_bio_covariates=5,
            n_tech_covariates=3,
        )

        batch_size = 32
        x = torch.randn(batch_size, 100)
        b_cov = torch.randn(batch_size, 5)
        t_cov = torch.randn(batch_size, 3)

        outputs = model.forward(x=x, b_cov=b_cov, t_cov=t_cov)
        loss_dict = model.loss(x=x, outputs=outputs, beta=1.0, beta_tc=1.0)

        # Check that loss components are present
        assert "loss" in loss_dict
        assert "reconstruction_loss" in loss_dict
        assert "kl_divergence" in loss_dict

        # Check that losses are scalars
        for key, value in loss_dict.items():
            assert isinstance(value, torch.Tensor)
            assert value.dim() == 0  # scalar tensor

    def test_gpu_compatibility(self):
        """Test that the model works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = inVAE(
            n_genes=100,
            n_bio_covariates=5,
            n_tech_covariates=3,
        ).cuda()

        batch_size = 32
        x = torch.randn(batch_size, 100).cuda()
        b_cov = torch.randn(batch_size, 5).cuda()
        t_cov = torch.randn(batch_size, 3).cuda()

        outputs = model.forward(x=x, b_cov=b_cov, t_cov=t_cov)
        loss_dict = model.loss(x=x, outputs=outputs, beta=1.0, beta_tc=1.0)

        # Check that outputs are on GPU
        for key, value in outputs.items():
            assert value.is_cuda

        # Check that losses are on GPU
        for key, value in loss_dict.items():
            assert value.is_cuda
