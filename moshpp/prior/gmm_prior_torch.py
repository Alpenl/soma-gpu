import math

import torch


class TorchMaxMixturePrior:
    def __init__(self, means, precs, weights):
        self.means = means
        self.precs = precs
        self.weights = weights

    def __call__(self, x):
        x = torch.as_tensor(x, dtype=self.means.dtype, device=self.means.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        diffs = x[:, None, :] - self.means[None, :, :]
        loglikelihoods = math.sqrt(0.5) * torch.matmul(diffs.unsqueeze(-2), self.precs.unsqueeze(0)).squeeze(-2)
        scores = (loglikelihoods**2).sum(dim=-1) - torch.log(self.weights).unsqueeze(0)
        min_component_idx = torch.argmin(scores, dim=1)

        batch_idx = torch.arange(x.shape[0], device=x.device)
        chosen_logls = loglikelihoods[batch_idx, min_component_idx]
        chosen_weights = self.weights[min_component_idx]
        extra = torch.sqrt(-torch.log(chosen_weights)).unsqueeze(-1)
        return torch.cat([chosen_logls, extra], dim=-1)


def prepare_gmm_prior(means, covars, weights):
    target_dtype = torch.as_tensor(means).dtype
    means = torch.as_tensor(means, dtype=torch.float64)
    covars = torch.as_tensor(covars, dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64)

    precs = torch.linalg.inv(covars)
    chols = torch.linalg.cholesky(precs)

    const = (2 * math.pi) ** (means.shape[1] / 2.0)
    sign, logabsdet = torch.linalg.slogdet(covars)
    if not torch.all(sign > 0):
        raise ValueError("Covariances must be positive definite.")
    log_sqrdets = 0.5 * logabsdet
    log_ratio = log_sqrdets - log_sqrdets.min()
    safe_weights = torch.clamp(weights, min=torch.finfo(weights.dtype).tiny)
    log_normalized_weights = torch.log(safe_weights) - math.log(const) - log_ratio
    min_log = math.log(torch.finfo(target_dtype).tiny)
    normalized_weights = torch.exp(torch.clamp(log_normalized_weights, min=min_log))

    return TorchMaxMixturePrior(
        means=means.to(target_dtype),
        precs=chols.to(target_dtype),
        weights=normalized_weights.to(target_dtype),
    )
