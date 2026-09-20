"""CNN-based tune estimator (JINST 21 (2026) P02049).

Consumes a ``FrontendFrame`` produced by the spectral preprocessing stage,
builds the two-channel CNN input, runs inference and optional Kalman
temporal smoothing, and emits a ``MeasurementResult(source='uar')``.

Model inference is decoupled via an injected ``predict_fn``:

    predict_fn(x_w: np.ndarray [2, L] float32) -> (q_nn, sigma_nn)

Two reference backends live at the bottom of this file:
``make_cuda_graph_backend`` (the CUDA-Graph ``InferenceEngine``) and
``make_cpu_backend`` (plain-torch fallback). Torch is imported lazily
inside the factories so this module can still be imported in environments
without CUDA or torch.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, Tuple

import numpy as np

from tune_pipeline.frames import FrontendFrame, MeasurementResult, QualityFlag
from tune_pipeline.mapping.mad import mad_normalize
from tune_pipeline.conventions import LOW_CUT_Q


# Trained weights are loaded from ``ckpt/production.pth`` next to this file
# when ``make_cpu_backend`` / ``make_cuda_graph_backend`` are called without an
# explicit ``model_path``.
DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ckpt", "production.pth")


PredictFn = Callable[[np.ndarray], Tuple[float, float]]


@dataclass(frozen=True)
class UAREstimatorConfig:
    """Immutable knobs for the CNN estimator."""

    tune_grid_size: int = 1024
    # low_cut_q comes from the torch-free single source
    # (tune_pipeline.conventions) so it cannot drift from the value used at
    # training time. mad_eps mirrors Config (config.py) by hand — keep those
    # two in sync.
    low_cut_q: float = LOW_CUT_Q
    mad_eps: float = 1e-8
    # Signal-channel preprocessing; it has to match what the checkpoint was
    # trained with. "feature_builder" (DEFAULT) = the shared
    # tune_pipeline.cnn_features FeatureBuilder (MAD z-score of the signal plus
    # a normalised weight channel). "mad_normalize" = MAD normalisation over the
    # valid region only (tune_pipeline.mapping.mad, eps 1e-8). The two differ by
    # ~16x in scale, so a mismatch degrades inference without raising an error.
    feature_mode: str = "feature_builder"
    initial_q: float = 0.25             # seed for the optional tracker

    # Optional Kalman temporal smoothing (KalmanTuneTracker). Set to False to
    # emit the raw per-frame network output without post-filtering.
    use_kalman_tracker: bool = False
    tracker_process_variance: float = 1e-6
    tracker_initial_uncertainty: float = 0.01
    # Optional robust (Huber) innovation gate of the tracker, in units of the
    # normalized innovation: |y|/√S > tracker_huber_c inflates R by (nu/c)².
    # Disabled by default (inf).
    tracker_huber_c: float = float("inf")

    def __post_init__(self):
        # Reject an unknown feature_mode rather than falling through to a
        # preprocessing path the checkpoint was not trained with.
        valid = ("feature_builder", "mad_normalize")
        if self.feature_mode not in valid:
            raise ValueError(
                f"feature_mode must be one of {valid}, got {self.feature_mode!r}"
            )


@dataclass
class UARHistoryState:
    """Append-only per-frame histories."""

    q: Deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    sigma: Deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    failed: Deque[bool] = field(default_factory=lambda: deque(maxlen=2048))

    def clear(self) -> None:
        self.q.clear()
        self.sigma.clear()
        self.failed.clear()


@dataclass
class UAREstimatorState:
    """Mutable per-run state."""

    tracker: object = None  # KalmanTuneTracker or None

    # Last-frame intermediates published for callers that want both the
    # raw NN output and the tracker-filtered MeasurementResult without
    # re-running preprocess / predict. Populated by ``process()``.
    last_raw_q: Optional[float] = None
    last_raw_sigma: Optional[float] = None
    last_x_w: Optional[np.ndarray] = None

    def clear(self, cfg: UAREstimatorConfig) -> None:
        self.last_raw_q = None
        self.last_raw_sigma = None
        self.last_x_w = None
        if self.tracker is not None:
            # Re-seed the tracker by rebuilding in place — clearer than
            # calling tracker.reset() because we want both state mean
            # and covariance reseeded from cfg in one shot.
            from uar_helpers import KalmanTuneTracker
            self.tracker = KalmanTuneTracker(
                process_variance=cfg.tracker_process_variance,
                initial_tune=cfg.initial_q,
                initial_uncertainty=cfg.tracker_initial_uncertainty,
                huber_c=cfg.tracker_huber_c,
            )


def _build_state(cfg: UAREstimatorConfig) -> UAREstimatorState:
    if not cfg.use_kalman_tracker:
        return UAREstimatorState(tracker=None)
    from uar_helpers import KalmanTuneTracker
    return UAREstimatorState(
        tracker=KalmanTuneTracker(
            process_variance=cfg.tracker_process_variance,
            initial_tune=cfg.initial_q,
            initial_uncertainty=cfg.tracker_initial_uncertainty,
            huber_c=cfg.tracker_huber_c,
        )
    )


class UAREstimator:
    """FrontendFrame → MeasurementResult via MAD + CNN inference.

    ``predict_fn`` is injected by the caller so this class does not
    pull in PyTorch / CUDA at import time; an environment without
    torch can still import ``UAREstimator`` to inspect its contract.
    """

    def __init__(
        self,
        cfg: UAREstimatorConfig,
        predict_fn: PredictFn,
        state: Optional[UAREstimatorState] = None,
        history: Optional[UARHistoryState] = None,
    ):
        self.cfg = cfg
        self._predict = predict_fn
        self.state = state if state is not None else _build_state(cfg)
        self.history = history if history is not None else UARHistoryState()


    def preprocess(self, frame: FrontendFrame) -> np.ndarray:
        """FrontendFrame → 2-channel [2, L] float32 tensor for the CNN.

        ``feature_mode="feature_builder"``: the shared ``FeatureBuilder``
        builds both channels (MAD z-score of the signal, weight normalised
        by its maximum).

        ``feature_mode="mad_normalize"``:
          1. Copy ``frame.mapped_psd`` to float32 (MAD mutates in place).
          2. MAD-normalise the signal channel (valid region only).
          3. Normalise the weight channel to [0, 1] by its own max.
          4. Stack.
        """
        cfg = self.cfg
        L = cfg.tune_grid_size

        x_linear = np.asarray(frame.mapped_psd, dtype=np.float32).reshape(-1).copy()
        if x_linear.shape[0] != L:
            raise ValueError(
                f"mapped_psd length {x_linear.shape[0]} does not match "
                f"tune_grid_size {L}"
            )

        if cfg.feature_mode == "feature_builder":
            # Shared FeatureBuilder path — the same channel construction used
            # at training time. mapped_psd/weight_sum are already on the q-grid,
            # so feed them straight in (NO re-map). Returns the same [2, L] the
            # CNN saw at train time (≈16x lower scale than the mad_normalize
            # branch below).
            import torch
            from uar_helpers import _feature_builder
            w_lin = np.asarray(frame.weight_sum, dtype=np.float32).reshape(-1)
            X = _feature_builder(cfg.low_cut_q).build(
                torch.from_numpy(np.ascontiguousarray(x_linear.reshape(1, -1))),
                torch.from_numpy(np.ascontiguousarray(w_lin.reshape(1, -1))),
            )
            return X[0].numpy().astype(np.float32)

        # MAD-normalise the signal channel (robust centring + scaling on
        # the valid region), matching the training-time preprocessing of a
        # checkpoint trained in this mode.
        #
        # Source the cutoff index from the frame's ``valid_mask`` — the
        # spectral preprocessing stage is the q-domain authority. Under the
        # convention ``valid_mask = (q_grid > low_cut_q)`` the first-True
        # index is numerically identical to ``ceil(cfg.low_cut_q * L / 0.5)``;
        # consuming the mask explicitly removes the dependence on
        # ``cfg.low_cut_q`` matching the upstream setting at runtime.
        # Fall back to the config-derived value when no mask is provided,
        # so a synthesised FrontendFrame without ``valid_mask`` still works.
        vm = getattr(frame, "valid_mask", None)
        if vm is not None and len(vm) == L and bool(np.any(vm)):
            valid_start = int(np.argmax(np.asarray(vm, dtype=bool)))
        else:
            valid_start = int(np.ceil(cfg.low_cut_q * L / 0.5))
        valid_start = max(0, min(valid_start, L))
        x_linear = mad_normalize(x_linear, valid_start, L, np.float32(cfg.mad_eps))

        w_frame = np.asarray(frame.weight_sum, dtype=np.float32).reshape(-1)
        w_max = np.float32(w_frame.max())
        w_mapped = w_frame / (w_max + np.float32(1e-12))

        return np.stack((x_linear, w_mapped), axis=0)


    def process(self, frame: FrontendFrame) -> MeasurementResult:
        """Run one frame through preprocessing, inference and the optional tracker.

        Frames flagged ``FEW_PEAKS`` upstream short-circuit to a
        ``failed=True`` result with the last q replayed: a mapped PSD that the
        spectral preprocessing stage declared unusable cannot produce a
        meaningful CNN prediction.
        """
        if int(frame.quality_flags) & int(QualityFlag.FEW_PEAKS):
            last_q = (
                self.history.q[-1] if len(self.history.q) > 0 else self.cfg.initial_q
            )
            last_sigma = (
                self.history.sigma[-1]
                if len(self.history.sigma) > 0
                else self.cfg.tracker_initial_uncertainty
            )
            self.history.q.append(last_q)
            self.history.sigma.append(last_sigma)
            self.history.failed.append(True)
            # Mirror the replay onto the published intermediates so a
            # caller reading state.last_raw_* on every frame never
            # sees stale prior-frame values. last_x_w is None on this
            # branch — no preprocess() was run on the degenerate input.
            self.state.last_raw_q = last_q
            self.state.last_raw_sigma = last_sigma
            self.state.last_x_w = None
            return MeasurementResult(q=last_q, sigma=last_sigma, failed=True, source="uar")

        x_w = self.preprocess(frame)
        q_nn, sigma_nn = self._predict(x_w)

        # Publish intermediates so callers can read the raw NN output
        # and the preprocessed tensor without re-running preprocess /
        # predict.
        self.state.last_raw_q = float(q_nn)
        self.state.last_raw_sigma = float(sigma_nn)
        self.state.last_x_w = x_w

        if self.state.tracker is not None:
            q_out, sigma_out = self.state.tracker.update(float(q_nn), float(sigma_nn))
        else:
            q_out, sigma_out = float(q_nn), float(sigma_nn)

        self.history.q.append(q_out)
        self.history.sigma.append(sigma_out)
        self.history.failed.append(False)
        return MeasurementResult(q=q_out, sigma=sigma_out, failed=False, source="uar")


# ---------------------------------------------------------------------
# Reference backends
# ---------------------------------------------------------------------


def make_cpu_backend(model_path: "str | None" = None, cfg=None) -> PredictFn:
    """Plain-torch CPU inference backend.

    Use when CUDA is unavailable or determinism beats latency. Not as
    fast as ``InferenceEngine``'s CUDA Graph path — expect ms-range
    per-frame latency — but it lets the full pipeline run end-to-end
    in a CPU-only environment.

    ``model_path`` defaults to the trained checkpoint (``DEFAULT_CKPT``)
    when ``None``. ``cfg`` is the ``Config`` instance used at training time
    (``config.py``); forwarded to ``TuneMeasurementCNN`` so the model
    architecture matches the checkpoint, and defaults to a fresh
    ``Config()`` when ``None``.
    """
    import torch
    from models import TuneMeasurementCNN
    from config import Config

    if model_path is None:
        model_path = DEFAULT_CKPT
    if cfg is None:
        cfg = Config()

    model = TuneMeasurementCNN(cfg)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to("cpu")

    def predict(x_w: np.ndarray) -> Tuple[float, float]:
        with torch.no_grad():
            if x_w.ndim == 2:
                x = torch.from_numpy(x_w).unsqueeze(0)  # [1, 2, L]
            else:
                x = torch.from_numpy(x_w)
            q, sigma = model(x)
            return float(q[0]), float(sigma[0])

    return predict


def make_cuda_graph_backend(model_path: "str | None" = None, cfg=None) -> PredictFn:
    """Wrap ``InferenceEngine`` (inference.py) into the ``PredictFn`` contract.

    Deferred import so importing ``uar_estimator`` does not require
    CUDA. Any caller on a GPU system hands the resulting callable to
    ``UAREstimator``. ``model_path`` defaults to the trained checkpoint
    (``DEFAULT_CKPT``) and ``cfg`` to a fresh ``Config()`` when ``None``.
    """
    from inference import InferenceEngine
    from config import Config

    if model_path is None:
        model_path = DEFAULT_CKPT
    if cfg is None:
        cfg = Config()

    engine = InferenceEngine(model_path=model_path, cfg=cfg)
    return engine.predict
