import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.prior.gmm_prior_torch import prepare_gmm_prior


def test_prepare_gmm_prior_matches_manual_residual_construction():
    means = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    covars = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.6, 0.4], dtype=torch.float32)
    x = torch.tensor([[0.1, -0.2]], dtype=torch.float32)

    prior = prepare_gmm_prior(means, covars, weights)
    residual = prior(x)

    precs = torch.linalg.inv(covars)
    chols = torch.linalg.cholesky(precs)
    sqrdets = torch.sqrt(torch.linalg.det(covars))
    const = (2 * math.pi) ** (means.shape[1] / 2.0)
    normalized_weights = weights / (const * (sqrdets / sqrdets.min()))

    logls = []
    scores = []
    for mean, chol, weight in zip(means, chols, normalized_weights):
        logl = math.sqrt(0.5) * ((x[0] - mean) @ chol)
        logls.append(logl)
        scores.append((logl**2).sum() - torch.log(weight))

    expected_idx = int(torch.argmin(torch.stack(scores)))
    expected = torch.cat(
        [
            logls[expected_idx],
            torch.sqrt(-torch.log(normalized_weights[expected_idx])).reshape(1),
        ]
    )

    assert residual.shape == (1, 3)
    assert torch.allclose(residual[0], expected.to(residual.dtype), atol=1e-6)


def test_prepare_gmm_prior_returns_batch_residuals():
    means = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    covars = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    weights = torch.tensor([1.0], dtype=torch.float32)
    x = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float32)

    prior = prepare_gmm_prior(means, covars, weights)
    residual = prior(x)

    assert residual.shape == (2, 3)


def test_prepare_gmm_prior_keeps_weights_finite_for_high_dimensional_covariances():
    means = torch.zeros(2, 69, dtype=torch.float32)
    covars = torch.stack(
        [
            torch.eye(69, dtype=torch.float32) * 1e-2,
            torch.eye(69, dtype=torch.float32) * 2e-2,
        ]
    )
    weights = torch.tensor([0.6, 0.4], dtype=torch.float32)

    prior = prepare_gmm_prior(means, covars, weights)

    assert torch.isfinite(prior.weights).all()
    assert (prior.weights > 0).all()
    assert prior.weights.dtype == weights.dtype
