"""
Model architectures for Schottky Tune Net (Attention Pooling)
Direct position-aware peak localization using attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config


class DWConv1dBlock(nn.Module):
    """
    Depthwise separable convolution residual block.

    Combines depthwise convolution (spatial) with pointwise convolution (channel mixing)
    for efficient parameter usage while maintaining expressiveness.
    """

    def __init__(self, C: int, k: int, gn_groups: int):
        super().__init__()
        self.dw = nn.Conv1d(C, C, kernel_size=k, padding=k // 2, groups=C, bias=False)
        self.pw = nn.Conv1d(C, C, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=C)
        self.gn2 = nn.GroupNorm(num_groups=gn_groups, num_channels=C)

    def forward(self, x):
        h = F.gelu(self.gn1(self.dw(x)))
        h = self.gn2(self.pw(h))
        return x + F.gelu(h)


class Conv1dBlock(nn.Module):
    """
    Standard convolution residual block.

    Uses two sequential convolutions with GroupNorm and residual connection.
    """

    def __init__(self, C: int, k: int, gn_groups: int):
        super().__init__()
        self.c1 = nn.Conv1d(C, C, kernel_size=k, padding=k // 2, bias=False)
        self.c2 = nn.Conv1d(C, C, kernel_size=k, padding=k // 2, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=C)
        self.gn2 = nn.GroupNorm(num_groups=gn_groups, num_channels=C)

    def forward(self, x):
        h = F.gelu(self.gn1(self.c1(x)))
        h = self.gn2(self.c2(h))
        return x + F.gelu(h)


class TuneMeasurementCNN(nn.Module):
    """
    Attention-based CNN with SEPARATE BRANCHES for mean and uncertainty prediction.

    Design points:
      - Learnable scaling gate for the uncertainty output
      - Attention entropy included in difficulty assessment
      - Independent optimization paths for position vs. uncertainty

    Architecture:
      Input [B, 2, L] → Shared early layers → Split into:
        - Mean branch: Position-focused (precise localization)
        - Uncertainty branch: Context-focused (difficulty assessment)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        C = cfg.base_ch
        k = cfg.kernel_size
        k_uncertainty = cfg.uncertainty_kernel_size
        gnG = cfg.gn_groups

        Block = (lambda C, k: DWConv1dBlock(C, k, gnG)) if cfg.use_depthwise else (
            lambda C, k: Conv1dBlock(C, k, gnG))

        # ===== SHARED EARLY LAYERS (Basic Feature Extraction) =====

        k_fine = 3
        k_coarse = 15

        self.fine_stem = nn.Sequential(
            nn.Conv1d(cfg.in_ch, C, kernel_size=k_fine, padding=k_fine // 2, bias=False),
            nn.GroupNorm(num_groups=gnG, num_channels=C),
            nn.GELU()
        )

        self.coarse_stem = nn.Sequential(
            nn.Conv1d(cfg.in_ch, C, kernel_size=k_coarse, padding=k_coarse // 2, bias=False),
            nn.GroupNorm(num_groups=gnG, num_channels=C),
            nn.GELU()
        )

        # Initial fusion of multi-scale features
        self.early_fusion = nn.Sequential(
            nn.Conv1d(2 * C, C, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gnG, num_channels=C),
            nn.GELU()
        )

        # One shared processing block
        self.shared_body = nn.Sequential(*[Block(C, k) for _ in range(1)])

        # ===== MEAN BRANCH (Position-Focused) =====

        self.mean_branch = nn.Sequential(*[Block(C, k) for _ in range(cfg.depth // 2)])

        # Dilated convolution for mean branch
        mean_tails = []
        for d in (2, 4, 8):
            mean_tails += [
                nn.Conv1d(C, C, kernel_size=3, padding=d, dilation=d,
                          groups=(C if cfg.use_depthwise else 1), bias=False),
                nn.GroupNorm(num_groups=gnG, num_channels=C), nn.GELU(),
                nn.Conv1d(C, C, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gnG, num_channels=C), nn.GELU()
            ]
        self.mean_tail = nn.Sequential(*mean_tails)

        # Attention projection for mean
        self.mean_attention_proj = nn.Sequential(
            nn.Conv1d(C, C // 2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gnG // 2, num_channels=C // 2),
            nn.GELU(),
            nn.Conv1d(C // 2, 1, kernel_size=1)
        )

        # ===== UNCERTAINTY BRANCH (Context-Focused) =====

        Block_uncertainty = (lambda C: DWConv1dBlock(C, k_uncertainty, gnG)) if cfg.use_depthwise else (
            lambda C: Conv1dBlock(C, k_uncertainty, gnG))

        self.uncertainty_branch = nn.Sequential(*[Block_uncertainty(C) for _ in range(cfg.depth // 2)])

        # Larger dilation for uncertainty
        uncertainty_tails = []
        for d in (4, 8, 16):
            uncertainty_tails += [
                nn.Conv1d(C, C, kernel_size=3, padding=d, dilation=d,
                          groups=(C if cfg.use_depthwise else 1), bias=False),
                nn.GroupNorm(num_groups=gnG, num_channels=C), nn.GELU(),
                nn.Conv1d(C, C, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gnG, num_channels=C), nn.GELU()
            ]
        self.uncertainty_tail = nn.Sequential(*uncertainty_tails)

        # Global statistics for uncertainty (4 features; the entropy indicator is appended separately)
        self.uncertainty_global_stats = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(C, C // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(C // 4, 4, kernel_size=1)  # 4 spatial features
        )

        # Attention projection for uncertainty
        self.uncertainty_attention_proj = nn.Sequential(
            nn.Conv1d(C, C // 2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gnG // 2, num_channels=C // 2),
            nn.GELU(),
            nn.Conv1d(C // 2, 1, kernel_size=1)
        )

        # ===== LEARNABLE UNCERTAINTY SCALING GATE =====
        # Takes 5 difficulty indicators (4 spatial + 1 entropy) and outputs a scaling factor

        self.uncertainty_scaling_gate = nn.Sequential(
            nn.LayerNorm(5),  # Normalize the 5 input features
            nn.Linear(5, 1),  # Learnable combination → log-scale factor
            nn.Softplus()  # Ensure positive scaling (output ≈ 1.0 initially)
        )

        # ===== SHARED COMPONENTS =====

        # Q-grid for soft-argmax
        L = cfg.n_points_per_frame
        q_grid = torch.arange(L, dtype=torch.float32) * (0.5 / L)
        self.register_buffer('q_grid', q_grid)

        self.tau = cfg.tau
        self.kappa = cfg.kappa

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with dual-branch architecture and learnable uncertainty scaling.

        Args:
            x: Input tensor [B, 2, L]

        Returns:
            Tuple of (q_pred, sigma_pred)
        """
        # ===== SHARED EARLY FEATURE EXTRACTION =====
        f_fine = self.fine_stem(x)
        f_coarse = self.coarse_stem(x)
        f_fused = torch.cat([f_fine, f_coarse], dim=1)
        f_shared = self.early_fusion(f_fused)
        f_shared = self.shared_body(f_shared)

        # ===== MEAN BRANCH (Position Prediction) =====
        f_mean = self.mean_branch(f_shared)
        f_mean = self.mean_tail(f_mean)

        # Generate attention logits for mean
        mean_attn_logits = self.mean_attention_proj(f_mean).squeeze(1)  # [B, L]
        mean_attn_logits = torch.nan_to_num(mean_attn_logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply temperature and masking
        tau = self.tau
        logits_scaled = (mean_attn_logits / tau).clamp_max(20.0)

        # Low-frequency mask
        mask = (self.q_grid >= self.cfg.low_cut_q)
        neg_large = torch.finfo(logits_scaled.dtype).min
        logits_scaled = logits_scaled.masked_fill(~mask[None, :], neg_large)

        # Attention weights and prediction
        attn_weights = F.softmax(logits_scaled, dim=-1)  # [B, L]
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        q_pred = (attn_weights * self.q_grid[None, :]).sum(dim=-1)  # [B]

        # ===== UNCERTAINTY BRANCH (Difficulty Assessment) =====
        f_uncertainty = self.uncertainty_branch(f_shared)
        f_uncertainty = self.uncertainty_tail(f_uncertainty)  # [B, C, L]

        # Extract global spatial statistics (4 features)
        global_stats_spatial = self.uncertainty_global_stats(f_uncertainty).squeeze(-1)  # [B, 4]

        # Generate softer attention logits for uncertainty
        uncertainty_attn_logits = self.uncertainty_attention_proj(f_uncertainty).squeeze(1)  # [B, L]
        uncertainty_attn_logits = torch.nan_to_num(uncertainty_attn_logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Softer temperature for uncertainty
        kappa = self.kappa
        logits_soft = (uncertainty_attn_logits / (tau * kappa)).clamp_max(20.0)
        logits_soft = logits_soft.masked_fill(~mask[None, :], neg_large)

        attn_soft = F.softmax(logits_soft, dim=-1)  # [B, L]

        # ===== COMPUTE ATTENTION ENTROPY (5th difficulty indicator) =====
        # Higher entropy → more ambiguous → higher uncertainty
        # Shannon entropy: H = -sum(p * log(p))
        attn_entropy = -(attn_soft * torch.log(attn_soft + 1e-12)).sum(dim=-1, keepdim=True)  # [B, 1]

        # Normalize entropy to [0, 1] range for stability
        # Max possible entropy for uniform distribution over L bins: log(L)
        L = self.q_grid.shape[0]
        max_entropy = np.log(L)
        attn_entropy_normalized = attn_entropy / max_entropy  # [B, 1]

        # ===== COMBINE ALL DIFFICULTY INDICATORS =====
        # 5 features: [4 spatial stats, 1 entropy]
        difficulty_features = torch.cat([global_stats_spatial, attn_entropy_normalized], dim=-1)  # [B, 5]

        # ===== LEARNABLE SCALING GATE =====
        # Transform difficulty indicators into a positive scaling factor
        # Initial behavior should be close to 1.0 (minimal modification)
        scaling_factor = self.uncertainty_scaling_gate(difficulty_features).squeeze(-1)  # [B]

        # ===== VARIANCE-BASED BASE UNCERTAINTY =====
        q_mean_soft = (attn_soft * self.q_grid[None, :]).sum(dim=-1).detach()
        q_variance = (attn_soft * (self.q_grid[None, :] - q_mean_soft[:, None]) ** 2).sum(dim=-1)
        sigma_base = torch.sqrt(q_variance + 1e-12)  # [B]

        # ===== MODULATE WITH LEARNED SCALING =====
        sigma_pred = sigma_base * scaling_factor

        # Physical clamping
        q_pred = q_pred.clamp(0.0, 0.5)
        sigma_pred = sigma_pred.clamp(1e-6, 0.2)

        return q_pred, sigma_pred
