"""Helpers for the CNN tune estimator.

Holds the cached shared ``FeatureBuilder`` that builds the two-channel CNN
input, and ``KalmanTuneTracker``, the scalar Kalman temporal smoother that
consumes the network's calibrated per-frame sigma as its measurement-noise
estimate.
"""

from typing import Tuple

import numpy as np


# Shared-FeatureBuilder cache (keyed by low_cut_q) for the inference path.
_FEATURE_BUILDERS: dict = {}


def _feature_builder(low_cut_q: float):
    fb = _FEATURE_BUILDERS.get(low_cut_q)
    if fb is None:
        from tune_pipeline.cnn_features import FeatureBuilder, FeatureConfig
        fb = FeatureBuilder(FeatureConfig(low_cut_q=low_cut_q))
        _FEATURE_BUILDERS[low_cut_q] = fb
    return fb


class KalmanTuneTracker:
    """Scalar adaptive Kalman filter for betatron tune tracking.

    Random-walk model (F=1, H=1) with closed-form scalar updates. The
    per-frame measurement noise R is the square of the network's calibrated
    ``sigma_nn``, so frames the network reports as uncertain automatically
    receive a smaller Kalman gain; this is what makes the temporal smoothing
    adaptive instead of fixed-bandwidth.

    An optional robust (Huber) innovation gate is available and disabled by
    default (``huber_c = inf``). When enabled, a normalized innovation
    |y|/√S above ``huber_c`` inflates R for that frame::

        nu = |z - x_pred| / √S
        if nu > huber_c:
            R_eff = R * (nu / huber_c)²
            S_eff = P_pred + R_eff
        K = P_pred / S_eff
    """

    def __init__(self,
                 process_variance: float = 1e-6,
                 initial_tune: float = 0.25,
                 initial_uncertainty: float = 0.01,
                 huber_c: float = float("inf")):
        self.Q = float(process_variance)
        self.huber_c = float(huber_c)

        self.state_mean = np.array([float(initial_tune)], dtype=np.float32)
        self.state_cov = np.array([[float(initial_uncertainty) ** 2]], dtype=np.float32)

        self._r_min = 1e-12
        self._p_min = 1e-12
        self._p_max = 0.25 ** 2

    def update(self, q_nn: float, sigma_nn: float) -> Tuple[float, float]:
        """Update the filter with a new network measurement.

        ``sigma_nn`` is the network's per-frame uncertainty, used as the
        measurement standard deviation.

        Returns ``(q_filtered, sigma_filtered)``.
        """
        x = float(self.state_mean[0])
        P = float(self.state_cov[0, 0])

        r = float(sigma_nn)
        R = max(r * r, self._r_min)

        x_pred = x
        P_pred = P + self.Q

        z = float(q_nn)
        y = z - x_pred
        S = P_pred + R
        if S <= 0.0:
            S = self._r_min

        # Optional Huber gate: inflate R for large innovations (no-op at huber_c = inf).
        nu = abs(y) / np.sqrt(S)
        if nu > self.huber_c:
            R_eff = R * (nu / self.huber_c) ** 2
            S_eff = P_pred + R_eff
        else:
            S_eff = S

        K = P_pred / S_eff

        x_new = x_pred + K * y
        P_new = (1.0 - K) * P_pred
        P_new = min(max(P_new, self._p_min), self._p_max)

        x_new = min(max(x_new, 0.0), 0.5)

        self.state_mean[0] = x_new
        self.state_cov[0, 0] = P_new

        return float(x_new), float(np.sqrt(P_new))

    def reset(self, initial_tune: float = 0.25, initial_uncertainty: float = 0.01):
        """Reset filter to initial state."""
        self.state_mean = np.array([float(initial_tune)], dtype=np.float32)
        self.state_cov = np.array([[float(initial_uncertainty) ** 2]], dtype=np.float32)
