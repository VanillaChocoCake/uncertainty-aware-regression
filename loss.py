"""Uncertainty-calibrating training objective for the tune-regression CNN.

The network outputs a tune estimate and a per-frame sigma; this loss is what
makes that sigma a calibrated error scale rather than an arbitrary number,
which is the precondition for using it as the measurement noise of the
downstream Kalman filter.
"""

import torch
import torch.nn as nn


class LaplaceNLL(nn.Module):
    """
    Laplace Negative Log-Likelihood (NLL) Loss.

    This loss function calibrates uncertainty predictions by penalizing:
        - Large errors with small sigma (overconfidence)
        - Small errors with large sigma (underconfidence)

    The Laplace NLL forces the model to learn:
        Large errors → Large σ (to reduce error/σ term)
        Small errors → Small σ (to reduce log(σ) term)

    This ensures proper uncertainty calibration for downstream applications
    (e.g., Kalman filtering).
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps                          # Numerical stability epsilon

    def forward(self, q_pred, sigma_pred, q_true):
        err = torch.abs(q_pred - q_true)

        # Clamp sigma to avoid division by zero
        sigma = sigma_pred.clamp(min=self.eps)

        # Laplace NLL: |error|/σ + log(σ)
        # Two competing terms:
        #   - |error|/σ: Pushes σ to be large when errors are large
        #   - log(σ):    Penalizes large σ, pushes it to be small
        # Balance is achieved when σ matches the true error magnitude
        nll = (err / sigma + torch.log(sigma)).mean()

        return {
            'loss': nll,
            'mae_loss': err.mean(),
            'sigma_mean': sigma.mean().item()
        }
