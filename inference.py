"""Real-time inference engine for the tune-regression CNN.

Wraps the trained model in a captured CUDA Graph so that a single
preprocessed frame can be evaluated with almost no launch overhead, which is
what keeps the per-frame latency inside the budget of a real-time control
loop. The engine only runs the network; preprocessing and the Kalman
smoothing live in ``uar_estimator`` / ``uar_helpers``.
"""

import numpy as np
import torch
from typing import Tuple
from config import Config
from models import TuneMeasurementCNN


class InferenceEngine:
    """
    CUDA Graph-based inference for minimum per-frame latency

    A CUDA Graph records the whole forward pass once and replays it with
    almost no CPU-side launch overhead.

    Limitation: the batch size is fixed at capture time (1 here)
    """

    def __init__(self,
                 model_path: str,
                 cfg: Config):
        """
        Initialize CUDA Graph engine

        Parameters
        ----------
        model_path : str
            Path to trained model
        cfg : Config
            Configuration
        """
        if cfg.device != 'cuda':
            raise ValueError("CUDA Graphs require CUDA device")

        self.cfg = cfg

        self.model = TuneMeasurementCNN(cfg).to(cfg.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        self.model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        self.model.eval()

        # Static tensors for CUDA graph
        L = cfg.n_points_per_frame
        self.static_input = torch.zeros(1, 2, L, device='cuda', dtype=torch.float32)
        self.h_in = torch.empty((1, 2, L), dtype=torch.float32, pin_memory=True)  # pinned host buffer
        self.static_output_q = torch.zeros(1, device='cuda', dtype=torch.float32)
        self.static_output_sigma = torch.zeros(1, device='cuda', dtype=torch.float32)

        # Warm-up
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(s)

        # Capture CUDA graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                self.static_output_q, self.static_output_sigma = self.model(self.static_input)

    def predict(self, X_preprocessed: np.ndarray) -> Tuple[float, float]:
        """
        Predict using CUDA graph replay
        X_preprocessed: numpy array with shape [2, L] or [1, 2, L], float32
        """

        # 1) Copy into the pinned host buffer (a host-to-host copy, so
        #    non_blocking would buy nothing here).
        if X_preprocessed.ndim == 2:  # [2, L]
            # Matching dtype and memory layout avoids an extra copy, so pass
            # float32 C-contiguous data in from upstream.
            self.h_in[0].copy_(torch.from_numpy(X_preprocessed), non_blocking=False)
        else:  # [1, 2, L]
            self.h_in.copy_(torch.from_numpy(X_preprocessed), non_blocking=False)

        # 2) Non-blocking host-to-device copy into the reused static buffer.
        #    This copies the contents rather than rebinding the tensor, so the
        #    addresses baked into the captured graph stay valid.
        self.static_input.copy_(self.h_in, non_blocking=True)

        # 3) Replay the captured graph.
        self.graph.replay()

        return float(self.static_output_q[0]), float(self.static_output_sigma[0])
