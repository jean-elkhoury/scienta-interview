"""Tests for utility functions."""

import pytest
import torch
from scienta.utils import kl_divergence_gaussian


class TestUtils:
    """Test cases for utility functions."""

    def test_kl_divergence_gaussian(self):
        """Test KL divergence computation between two Gaussians."""
        # Test with simple case where q = p (should give 0)
        q_mean = torch.tensor([[0.0, 0.0]])  # 2D tensor
        q_log_var = torch.tensor([[0.0, 0.0]])  # var = 1
        p_mean = torch.tensor([[0.0, 0.0]])
        p_log_var = torch.tensor([[0.0, 0.0]])  # var = 1

        kl = kl_divergence_gaussian(q_mean, q_log_var, p_mean, p_log_var)

        # KL divergence should be 0 when distributions are identical
        assert torch.allclose(kl, torch.tensor([0.0]), atol=1e-6)

    def test_kl_divergence_gaussian_different_means(self):
        """Test KL divergence with different means."""
        q_mean = torch.tensor([[0.0, 0.0]])  # 2D tensor
        q_log_var = torch.tensor([[0.0, 0.0]])  # var = 1
        p_mean = torch.tensor([[1.0, 1.0]])
        p_log_var = torch.tensor([[0.0, 0.0]])  # var = 1

        kl = kl_divergence_gaussian(q_mean, q_log_var, p_mean, p_log_var)

        # KL divergence should be positive when means are different
        assert kl > 0
        # For unit variance, KL = 0.5 * (mean_diff^2) = 0.5 * 2 = 1
        expected_kl = torch.tensor([1.0])
        assert torch.allclose(kl, expected_kl, atol=1e-6)

    def test_kl_divergence_gaussian_different_vars(self):
        """Test KL divergence with different variances."""
        q_mean = torch.tensor([[0.0, 0.0]])  # 2D tensor
        q_log_var = torch.tensor([[0.0, 0.0]])  # var = 1
        p_mean = torch.tensor([[0.0, 0.0]])
        p_log_var = torch.tensor([[1.0, 1.0]])  # var = e

        kl = kl_divergence_gaussian(q_mean, q_log_var, p_mean, p_log_var)

        # KL divergence should be positive when variances are different
        assert kl > 0

    def test_kl_divergence_gaussian_batch(self):
        """Test KL divergence with batch of samples."""
        batch_size = 10
        q_mean = torch.randn(batch_size, 5)
        q_log_var = torch.randn(batch_size, 5)
        p_mean = torch.randn(batch_size, 5)
        p_log_var = torch.randn(batch_size, 5)

        kl = kl_divergence_gaussian(q_mean, q_log_var, p_mean, p_log_var)

        # Check output shape
        assert kl.shape == (batch_size,)
        # All values should be non-negative (KL divergence is always >= 0)
        assert torch.all(kl >= 0)

    def test_kl_divergence_gaussian_gpu(self):
        """Test KL divergence on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        q_mean = torch.tensor([[0.0, 0.0]]).cuda()  # 2D tensor
        q_log_var = torch.tensor([[0.0, 0.0]]).cuda()
        p_mean = torch.tensor([[1.0, 1.0]]).cuda()
        p_log_var = torch.tensor([[0.0, 0.0]]).cuda()

        kl = kl_divergence_gaussian(q_mean, q_log_var, p_mean, p_log_var)

        # Check that output is on GPU
        assert kl.is_cuda
        # Check that computation is correct
        expected_kl = torch.tensor([1.0]).cuda()
        assert torch.allclose(kl, expected_kl, atol=1e-6)
