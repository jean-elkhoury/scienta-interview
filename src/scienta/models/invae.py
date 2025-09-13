import torch
import torch.nn as nn
from torch.distributions import Normal
from scienta.utils import kl_divergence_gaussian


class MLP(nn.Module):
    """Basic MLP block."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_in
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, n_hidden))
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = n_hidden
        layers.append(nn.Linear(n_hidden, n_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class inVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_bio_covariates: int,
        n_tech_covariates: int,
        n_latent_inv: int = 30,
        n_latent_spur: int = 5,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_latent_inv = n_latent_inv
        self.n_latent_spur = n_latent_spur
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        # Shared encoder
        encoder_input_dim = n_genes + n_bio_covariates + n_tech_covariates
        self.shared_encoder_mlp = MLP(
            encoder_input_dim, n_hidden, n_hidden, n_layers, dropout_rate
        )

        # Latent mean and logvar
        self.mean_inv_encoder = nn.Linear(n_hidden, n_latent_inv)
        self.log_var_inv_encoder = nn.Linear(n_hidden, n_latent_inv)
        self.mean_spur_encoder = nn.Linear(n_hidden, n_latent_spur)
        self.log_var_spur_encoder = nn.Linear(n_hidden, n_latent_spur)

        # Decoder
        decoder_input_dim = n_latent_inv + n_latent_spur
        self.shared_decoder_mlp = MLP(
            decoder_input_dim, n_hidden, n_hidden, n_layers, dropout_rate
        )

        self.px_log_mean_decoder = nn.Linear(n_hidden, n_genes)
        self.px_log_disp_decoder = nn.Linear(n_hidden, n_genes)

        # Invariant prior
        self.prior_inv_mlp = MLP(
            n_bio_covariates, n_hidden, n_hidden, n_layers, dropout_rate
        )
        self.prior_inv_mean = nn.Linear(n_hidden, n_latent_inv)
        self.prior_inv_log_var = nn.Linear(n_hidden, n_latent_inv)

        # Spurious prior
        self.prior_spur_mlp = MLP(
            n_tech_covariates, n_hidden, n_hidden, n_layers, dropout_rate
        )
        self.prior_spur_mean = nn.Linear(n_hidden, n_latent_spur)
        self.prior_spur_log_var = nn.Linear(n_hidden, n_latent_spur)

    def sample_latent(self, mean: torch.Tensor, log_var: torch.Tensor):
        """Standard VAE reparameterization trick."""
        std = (
            torch.exp(0.5 * log_var) + 1e-4
        )  # numerical stability (from theislab code)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(
        self,
        x: torch.Tensor,
        b_cov: torch.Tensor,
        t_cov: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Get latent distribution parameters."""
        encoder_input = torch.cat((x, b_cov, t_cov), dim=1)
        q = self.shared_encoder_mlp(encoder_input)
        q_mean_inv = self.mean_inv_encoder(q)
        q_log_var_inv = self.log_var_inv_encoder(q)
        q_mean_spur = self.mean_spur_encoder(q)
        q_log_var_spur = self.log_var_spur_encoder(q)

        return {
            "q_mean_inv": q_mean_inv,
            "q_log_var_inv": q_log_var_inv,
            "q_mean_spur": q_mean_spur,
            "q_log_var_spur": q_log_var_spur,
        }

    def prior_spur(
        self,
        t_cov: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        p_spur = self.prior_spur_mlp(t_cov)
        p_mean_spur = self.prior_spur_mean(p_spur)
        p_log_var_spur = self.prior_spur_log_var(p_spur)
        return {
            "p_mean_spur": p_mean_spur,
            "p_log_var_spur": p_log_var_spur,
        }

    def prior_inv(
        self,
        b_cov: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        p_inv = self.prior_inv_mlp(b_cov)
        p_mean_inv = self.prior_inv_mean(p_inv)
        p_log_var_inv = self.prior_inv_log_var(p_inv)
        return {
            "p_mean_inv": p_mean_inv,
            "p_log_var_inv": p_log_var_inv,
        }

    def decode(
        self,
        z_concat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        px = self.shared_decoder_mlp(z_concat)
        px_log_mean = self.px_log_mean_decoder(px)
        px_log_disp = self.px_log_disp_decoder(px)
        return {
            "px_log_mean": px_log_mean,
            "px_log_disp": px_log_disp,
        }

    def forward(
        self,
        x: torch.Tensor,
        b_cov: torch.Tensor,
        t_cov: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent_params = self.encode(x, b_cov, t_cov)
        z_inv = self.sample_latent(
            latent_params["q_mean_inv"], latent_params["q_log_var_inv"]
        )
        z_spur = self.sample_latent(
            latent_params["q_mean_spur"], latent_params["q_log_var_spur"]
        )
        z_concat = torch.cat((z_inv, z_spur), dim=1)
        reconstructed_params = self.decode(z_concat)
        spur_prior_params = self.prior_spur(t_cov)
        inv_prior_params = self.prior_inv(b_cov)
        return (
            latent_params
            | reconstructed_params
            | spur_prior_params
            | inv_prior_params
            | {
                "z_concat": z_concat,
                "z_inv": z_inv,
                "z_spur": z_spur,
            }
        )

    def loss(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        beta: float,
        beta_tc: float,
    ):
        # your existing recon / KL parts
        disp = torch.exp(outputs["px_log_disp"])
        mean = torch.exp(outputs["px_log_mean"])
        recons_dist = Normal(loc=mean, scale=disp)
        log_likelihood = recons_dist.log_prob(x).sum(dim=-1)
        reconstruction_loss = -torch.mean(log_likelihood)

        kl_inv = kl_divergence_gaussian(
            outputs["q_mean_inv"],
            outputs["q_log_var_inv"],
            outputs["p_mean_inv"],
            outputs["p_log_var_inv"],
        )

        kl_spur = kl_divergence_gaussian(
            outputs["q_mean_spur"],
            outputs["q_log_var_spur"],
            outputs["p_mean_spur"],
            outputs["p_log_var_spur"],
        )

        kl_local = beta * torch.mean(kl_inv + kl_spur)

        total_correlation = beta_tc * self.total_correlation(outputs)
        total_loss = reconstruction_loss + kl_local + total_correlation

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_local,
            "total_correlation": total_correlation,
        }

    def total_correlation(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        z_inv = outputs["z_inv"]
        z_spur = outputs["z_spur"]
        z_joint = torch.cat([z_inv, z_spur], dim=1)

        q_mean_joint = torch.cat([outputs["q_mean_inv"], outputs["q_mean_spur"]], dim=1)
        q_log_var_joint = torch.cat(
            [outputs["q_log_var_inv"], outputs["q_log_var_spur"]], dim=1
        )
        q_std_joint = torch.exp(0.5 * q_log_var_joint)

        q_mean_inv = outputs["q_mean_inv"]
        q_log_var_inv = outputs["q_log_var_inv"]
        q_std_inv = torch.exp(0.5 * q_log_var_inv)

        q_mean_spur = outputs["q_mean_spur"]
        q_log_var_spur = outputs["q_log_var_spur"]
        q_std_spur = torch.exp(0.5 * q_log_var_spur)

        # Create Normal distributions
        q_joint = Normal(loc=q_mean_joint, scale=q_std_joint)
        q_inv = Normal(loc=q_mean_inv, scale=q_std_inv)
        q_spur = Normal(loc=q_mean_spur, scale=q_std_spur)

        # Compute log probs
        log_q_joint = q_joint.log_prob(z_joint).sum(dim=1)
        log_q_inv = q_inv.log_prob(z_inv).sum(dim=1)
        log_q_spur = q_spur.log_prob(z_spur).sum(dim=1)

        # TC = log q(z) - [log q(z_inv) + log q(z_spur)]
        tc_per_sample = log_q_joint - (log_q_inv + log_q_spur)
        total_correlation = torch.mean(tc_per_sample)
        return total_correlation
