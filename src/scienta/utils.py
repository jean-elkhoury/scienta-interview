import torch


def kl_divergence_gaussian(q_mean, q_log_var, p_mean, p_log_var):
    term1 = p_log_var - q_log_var
    term2 = (torch.exp(q_log_var) + (q_mean - p_mean).pow(2)) / torch.exp(p_log_var)
    kl = 0.5 * torch.sum(term1 + term2 - 1, dim=1)
    return kl
