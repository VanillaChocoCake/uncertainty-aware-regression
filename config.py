"""
Configuration settings for Schottky Tune Net (Attention Pooling)
"""

from dataclasses import dataclass
import torch

from tune_pipeline.conventions import LOW_CUT_Q


@dataclass
class Config:
    # ============= DATA =============
    n_points_per_frame: int = 1024          # Number of frequency bins (L)
    low_cut_q: float = LOW_CUT_Q            # single source: tune_pipeline.conventions.LOW_CUT_Q
    spurious_p: float = 0.21                # Probability of injecting a synthetic spurious peak into a training frame

    # ============= TRAINING =============
    batch_size: int = 128
    n_epochs: int = 100              # Number of training epochs
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 114514

    # ============= OPTIMIZATION =============
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_amp: bool = False

    # ============= MODEL ARCHITECTURE =============
    in_ch: int = 2                          # Input channels: [spectrum, weights]
    base_ch: int = 32                       # Base number of feature channels
    depth: int = 4                          # Number of convolutional blocks
    kernel_size: int = 7                    # Kernel size of the residual-block convolutions
    uncertainty_kernel_size: int = 21       # Kernel size of the uncertainty-branch convolutions
    use_depthwise: bool = True              # Use depthwise separable convolutions
    gn_groups: int = 4                     # GroupNorm number of groups

    # ============= OTHER =============
    mad_eps: float = 1e-8                   # MAD normalization numerical stability epsilon

    # ============= ATTENTION PARAMETERS =============
    tau: float = 0.8                        # Temperature for attention sharpness
    kappa: float = 2.0                      # Softening factor for uncertainty estimation (fixed)
