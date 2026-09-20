"""MAD (median absolute deviation) normalisation of the mapped PSD.

Robust per-frame centring and scaling of the tune-axis PSD, computed over the
valid region of the axis only, with an eps of 1e-8 on the scale.

This is the signal-channel preprocessing of the estimator's
``feature_mode="mad_normalize"`` path. The default
``feature_mode="feature_builder"`` path instead uses the shared
``tune_pipeline.cnn_features.FeatureBuilder``, which takes the median over the
full axis and uses an eps of 1e-20; the two therefore differ in scale and a
checkpoint must be used with the mode it was trained with.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def mad_normalize(x_linear: np.ndarray,
                  valid_start: int,
                  bins: int,
                  mad_eps: float = 1e-8) -> np.ndarray:
    """In-place MAD-based robust centring and scaling.

    Computes the median ``b0`` and MAD ``s0 = 1.4826 * median(|x - b0|)``
    on the valid region ``x_linear[valid_start:]``, then applies
    ``x_linear -= b0; x_linear *= 1 / (s0 + mad_eps)``. Invalid prefix
    bins are zeroed. Falls back to IQR / std when MAD is numerically
    degenerate.

    Parameters
    ----------
    x_linear : np.ndarray [L] float32
        MUTATED in place. Call with a copy if you need to preserve the
        input.
    valid_start : int
        First bin index considered "valid" (matches
        ``int(np.ceil(low_cut_q * bins / 0.5))`` at the call site).
    bins : int
        Grid size; used as the upper bound when ``valid_start >= bins``.
    mad_eps : float
        Numerical floor added to the MAD scale before division.

    Returns
    -------
    x_linear : np.ndarray
        The same array, normalized in place and returned for chaining.
    """
    if valid_start < bins:
        U = x_linear[valid_start:].copy()
    else:
        U = x_linear.copy()

    U.sort()
    n = len(U)
    if n % 2 == 0:
        b0 = (U[n // 2 - 1] + U[n // 2]) / 2.0
    else:
        b0 = U[n // 2]

    abs_dev = np.abs(x_linear[valid_start:] - b0) if valid_start < bins else np.abs(x_linear - b0)
    abs_dev_sorted = abs_dev.copy()
    abs_dev_sorted.sort()
    n_dev = len(abs_dev_sorted)
    if n_dev % 2 == 0:
        mad = (abs_dev_sorted[n_dev // 2 - 1] + abs_dev_sorted[n_dev // 2]) / 2.0
    else:
        mad = abs_dev_sorted[n_dev // 2]

    s0 = 1.4826 * mad

    if s0 < 1e-12:
        q25_idx = n // 4
        q75_idx = (3 * n) // 4
        q25 = U[q25_idx]
        q75 = U[q75_idx]
        iqr_std = (q75 - q25) / 1.349

        mean = np.mean(U)
        std = np.sqrt(np.mean((U - mean) ** 2))

        s0 = max(iqr_std, std, 1e-9)

    g = np.float32(1.0 / (s0 + mad_eps))
    x_linear -= b0
    x_linear *= g

    if valid_start > 0:
        x_linear[:valid_start] = 0.0

    return x_linear
