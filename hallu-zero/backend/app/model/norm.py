"""
Adaptive Pre-Norm (APN)
========================
Standard Pre-Norm:  out = x + sublayer(LayerNorm(x))
Adaptive Pre-Norm:  out = x + sublayer(LayerNorm(x) * alpha + beta)

Where alpha and beta are *learned per-token* scale and shift vectors,
predicted by a small MLP that reads the current token embedding.

Why it helps:
- Standard LayerNorm applies the same learned gamma/beta to ALL tokens.
- APN lets each token control its own normalization scale — tokens carrying
  factual claims can sharpen their representation differently from
  function words like "the" or "and".
- This is especially useful for hallucination reduction: factual tokens
  need high-confidence, well-separated representations.

Reference: inspired by DiT (Peebles & Xie 2022) adaptive layer norm
           and MAGNETO (Wang et al. 2022) sub-LN scaling.
"""

import torch
import torch.nn as nn
from typing import Tuple


class AdaptivePreNorm(nn.Module):
    """
    Adaptive Pre-Normalization layer.

    Args:
        d_model:      model dimension
        cond_dim:     conditioning signal dimension (defaults to d_model)
                      Can be a separate context vector (e.g. from JEPA encoder)
                      or the token itself (self-conditioned)
        eps:          LayerNorm epsilon
    """

    def __init__(
        self,
        d_model: int,
        cond_dim: int = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        cond_dim = cond_dim or d_model
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)

        # Predict (alpha, beta) from conditioning signal
        # alpha initialized to 1, beta to 0 → starts as standard LN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model, bias=True),
        )
        # Init: alpha→1, beta→0
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.constant_(self.adaLN_modulation[-1].bias[:d_model], 1.0)   # alpha
        nn.init.zeros_(self.adaLN_modulation[-1].bias[d_model:])            # beta

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, T, d_model) — token embeddings
            cond: (B, T, cond_dim) or (B, cond_dim) — conditioning signal
                  If None, x is used as self-condition.

        Returns:
            x_norm: normalized + modulated x, ready for sublayer
            alpha:  scale factors (for residual re-scaling if needed)
            beta:   shift factors
        """
        if cond is None:
            cond = x  # self-conditioned

        # If cond is (B, D), expand to (B, T, D)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, x.shape[1], -1)

        modulation = self.adaLN_modulation(cond)           # (B, T, 2*D)
        alpha, beta = modulation.chunk(2, dim=-1)           # each (B, T, D)

        x_norm = self.norm(x) * alpha + beta
        return x_norm, alpha, beta


class AdaptivePreNormLayer(nn.Module):
    """
    Drop-in replacement for standard Pre-Norm that wraps any sublayer.

    Usage:
        self.attn_norm = AdaptivePreNormLayer(d_model, sublayer=attn)
        out = self.attn_norm(x, cond=jepa_context)
    """

    def __init__(self, d_model: int, sublayer: nn.Module, cond_dim: int = None):
        super().__init__()
        self.apn = AdaptivePreNorm(d_model, cond_dim=cond_dim)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        x_norm, _, _ = self.apn(x, cond=cond)
        return x + self.sublayer(x_norm, **kwargs)
