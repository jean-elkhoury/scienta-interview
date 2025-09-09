import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class MLP(nn.Module):
    """Basic MLP block."""

    def __init__(self, n_in, n_out, n_hidden=128, n_layers=2, dropout_rate=0.1):
        super().__init__()
        layers = []
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
        n_input,
        n_bio_covariates,
        n_tech_covariates,
        n_latent_inv=30,
        n_latent_spur=5,
        n_hidden=128,
        n_layers=2,
        dropout_rate=0.1,
    ):
        super().__init__()

        # --- ENCODER ---
        # Takes concatenated input (X), biological covariates (B), and technical covariates (T)
        encoder_input_dim = n_input + n_bio_covariates + n_tech_covariates
        self.shared_encoder_mlp = MLP(
            encoder_input_dim, n_hidden, n_hidden, n_layers, dropout_rate
        )

        # Output layers for the posterior distribution q(Z|X, B, T)
        self.mean_inv_encoder = nn.Linear(n_hidden, n_latent_inv)
        self.log_var_inv_encoder = nn.Linear(n_hidden, n_latent_inv)
        self.mean_spur_encoder = nn.Linear(n_hidden, n_latent_spur)
        self.log_var_spur_encoder = nn.Linear(n_hidden, n_latent_spur)

        # --- DECODER ---
        # Takes concatenated latent variables (Z_invariant, Z_spurious)
        decoder_input_dim = n_latent_inv + n_latent_spur
        self.shared_decoder_mlp = MLP(
            decoder_input_dim, n_hidden, n_hidden, n_layers, dropout_rate
        )

        # Output layers for the Negative Binomial distribution parameters
        # We model log(mean) and log(dispersion) to ensure they are positive
        self.px_log_mean_decoder = nn.Linear(n_hidden, n_input)
        self.px_log_disp_decoder = nn.Linear(n_hidden, n_input)

        # --- PRIOR NETWORKS ---
        # These networks learn the parameters of the prior distributions
        # p(Z_invariant | B) and p(Z_spurious | T)

        # Invariant prior network
        self.prior_inv_mlp = MLP(
            n_bio_covariates, n_hidden, n_hidden, n_layers, dropout_rate
        )
        self.prior_inv_mean = nn.Linear(n_hidden, n_latent_inv)
        self.prior_inv_log_var = nn.Linear(n_hidden, n_latent_inv)

        # Spurious prior network
        self.prior_spur_mlp = MLP(
            n_tech_covariates, n_hidden, n_hidden, n_layers, dropout_rate
        )
        self.prior_spur_mean = nn.Linear(n_hidden, n_latent_spur)
        self.prior_spur_log_var = nn.Linear(n_hidden, n_latent_spur)

    def sample_latent(self, mean, log_var):
        """Standard VAE reparameterization trick."""
        std = (
            torch.exp(0.5 * log_var) + 1e-4
        )  # numerical stability (from theislab code)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, b_cov, t_cov):
        """
        Forward pass of the inVAE model.

        Args:
            x (torch.Tensor): Input data (e.g., gene counts).
            b_cov (torch.Tensor): One-hot encoded biological covariates.
            t_cov (torch.Tensor): One-hot encoded technical covariates.

        Returns:
            dict: A dictionary containing all necessary tensors for loss calculation.
        """
        # Encoding
        # Get parameters for the posterior distribution q(Z|X,B,T)
        encoder_input = torch.cat((x, b_cov, t_cov), dim=1)
        q = self.shared_encoder_mlp(encoder_input)
        q_mean_inv = self.mean_inv_encoder(q)
        q_log_var_inv = self.log_var_inv_encoder(q)
        q_mean_spur = self.mean_spur_encoder(q)
        q_log_var_spur = self.log_var_spur_encoder(q)

        # Sample in latent space
        # Sample latent variables using the reparameterization trick
        z_inv = self.sample_latent(q_mean_inv, q_log_var_inv)
        z_spur = self.sample_latent(q_mean_spur, q_log_var_spur)

        # Decoding
        # Reconstruct the input data from the latent samples
        z_concat = torch.cat((z_inv, z_spur), dim=1)
        px = self.shared_decoder_mlp(z_concat)
        px_log_mean = self.px_log_mean_decoder(px)
        px_log_disp = self.px_log_disp_decoder(px)

        # Prior parameters
        # Get parameters for the prior distributions p(Z_inv|B) and p(Z_spur|T)
        p_inv = self.prior_inv_mlp(b_cov)
        p_mean_inv = self.prior_inv_mean(p_inv)
        p_log_var_inv = self.prior_inv_log_var(p_inv)

        p_spur = self.prior_spur_mlp(t_cov)
        p_mean_spur = self.prior_spur_mean(p_spur)
        p_log_var_spur = self.prior_spur_log_var(p_spur)

        return {
            "px_log_mean": px_log_mean,
            "px_log_disp": px_log_disp,
            "q_mean_inv": q_mean_inv,
            "q_log_var_inv": q_log_var_inv,
            "q_mean_spur": q_mean_spur,
            "q_log_var_spur": q_log_var_spur,
            "p_mean_inv": p_mean_inv,
            "p_log_var_inv": p_log_var_inv,
            "p_mean_spur": p_mean_spur,
            "p_log_var_spur": p_log_var_spur,
        }

    def loss(self, x, outputs, beta=1.0):
        """
        Calculates the Evidence Lower Bound (ELBO) loss for the inVAE model.

        Args:
            x (torch.Tensor): The original input data.
            outputs (dict): The dictionary returned by the forward pass.
            beta (float): The weight for the KL divergence term.

        Returns:
            dict: A dictionary containing the total loss, reconstruction loss, and KL divergence.
        """
        # --- 1. Reconstruction Loss (Negative Log-Likelihood) ---
        # We use a Negative Binomial distribution for the reconstruction
        disp = torch.exp(outputs["px_log_disp"])
        mean = torch.exp(outputs["px_log_mean"])

        # More stable parameterization for NB with mean and dispersion
        # probs = mu / (mu + theta)
        # We use logits = log(probs / (1 - probs)) = log(mu / theta) = log_mu - log_theta
        # recons_dist = NegativeBinomial(total_count=disp, logits=mean.log() - disp.log())
        recons_dist = Normal(loc=mean, scale=disp)
        log_likelihood = recons_dist.log_prob(x).sum(dim=-1)
        reconstruction_loss = -torch.mean(log_likelihood)

        def kl_divergence_gaussian(q_mean, q_log_var, p_mean, p_log_var):
            term1 = p_log_var - q_log_var
            term2 = (torch.exp(q_log_var) + (q_mean - p_mean).pow(2)) / torch.exp(
                p_log_var
            )
            kl = 0.5 * torch.sum(term1 + term2 - 1, dim=1)
            return kl

        # KL for invariant
        kl_inv = kl_divergence_gaussian(
            outputs["q_mean_inv"],
            outputs["q_log_var_inv"],
            outputs["p_mean_inv"],
            outputs["p_log_var_inv"],
        )

        # KL for spurious
        kl_spur = kl_divergence_gaussian(
            outputs["q_mean_spur"],
            outputs["q_log_var_spur"],
            outputs["p_mean_spur"],
            outputs["p_log_var_spur"],
        )

        kl_local = torch.mean(kl_inv + kl_spur)
        # Total loss
        total_loss = reconstruction_loss + beta * kl_local

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_local,
        }


if __name__ == "__main__":
    # --- DEMONSTRATION OF MODEL INSTANTIATION, FORWARD, AND BACKWARD PASS ---

    # 1. Define model parameters
    N_INPUT_GENES = 50
    N_CELLS = 256  # Batch size
    N_BIO_COVARIATES = 1  # e.g., 5 different disease variants
    N_TECH_COVARIATES = 1  # e.g., 3 different sequencing batches

    # 2. Create dummy data
    # Gene expression data (raw counts)
    x_data = torch.randint(0, 1000, (N_CELLS, N_INPUT_GENES)).float()

    # Biological covariates (one-hot encoded)
    b_cov_indices = torch.randint(0, N_BIO_COVARIATES, (N_CELLS,))
    b_cov_data = F.one_hot(b_cov_indices, num_classes=N_BIO_COVARIATES).float()

    # Technical covariates (one-hot encoded)
    t_cov_indices = torch.randint(0, N_TECH_COVARIATES, (N_CELLS,))
    t_cov_data = F.one_hot(t_cov_indices, num_classes=N_TECH_COVARIATES).float()

    # 3. Instantiate the model
    model = inVAE(
        n_input=N_INPUT_GENES,
        n_bio_covariates=N_BIO_COVARIATES,
        n_tech_covariates=N_TECH_COVARIATES,
        n_latent_inv=10,  # Smaller latent space for demo
        n_latent_spur=3,
    )

    print("--- Model Instantiated ---")
    print(model)

    # 4. Perform a forward pass
    print("\n--- Testing Forward Pass ---")
    print(f"Input shape (genes): {x_data.shape}")
    print(f"Input shape (bio covariates): {b_cov_data.shape}")
    print(f"Input shape (tech covariates): {t_cov_data.shape}")

    outputs = model(x_data, b_cov_data, t_cov_data)

    print("\nForward pass successful. Output keys:")
    print(list(outputs.keys()))
    print(f"Shape of reconstructed mean: {outputs['px_log_mean'].shape}")

    # 5. Calculate loss
    loss_dict = model.loss(x_data, outputs)
    total_loss = loss_dict["loss"]

    print(f"\nCalculated loss: {total_loss.item():.4f}")

    # 6. Perform a backward pass
    print("\n--- Testing Backward Pass ---")
    # In a real training loop, you would zero the gradients first: optimizer.zero_grad()
    total_loss.backward()
    print("Backward pass executed successfully.")

    # Check if gradients are computed for a sample parameter
    sample_param = next(model.parameters())
    if sample_param.grad is not None:
        print("Gradients have been computed and stored in .grad attributes.")
    else:
        print("Gradients were not computed.")
