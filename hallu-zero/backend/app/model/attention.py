"""
Dynamic Query Attention (DQA)
==============================
Standard multi-head attention computes Q, K, V with fixed linear projections.
DQA makes the query projection *input-dependent* — the projection weights are
modulated by a lightweight gating network that reads the current token context.

This lets each token dynamically decide HOW to query the context, rather than
always using the same fixed projection — critical for factual grounding.

Architecture:
  x → LayerNorm → [K_proj, V_proj]          (standard)
  x → LayerNorm → gate_net → dynamic W_q    (dynamic)
  Q = x @ (W_q_base + gate * W_q_delta)
  Attention(Q, K, V) with scaled dot-product
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DynamicQueryAttention(nn.Module):
    """
    Multi-head attention with input-conditioned query projection.

    Args:
        d_model:    model dimension
        n_heads:    number of attention heads
        dropout:    attention dropout probability
        gate_rank:  rank of the low-rank delta for query modulation
                    (smaller = cheaper, default 16)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        gate_rank: int = 16,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = math.sqrt(self.d_head)
        self.gate_rank = gate_rank

        # Standard K, V projections (fixed)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Base query projection (fixed component)
        self.q_proj_base = nn.Linear(d_model, d_model, bias=False)

        # Dynamic delta: low-rank decomposition W_q_delta = A @ B
        # A: d_model → gate_rank,  B: gate_rank → d_model*d_model
        # To keep memory sane, we produce a per-head scale vector instead
        # of a full d_model×d_model matrix.
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, gate_rank, bias=False),
            nn.SiLU(),
            nn.Linear(gate_rank, n_heads, bias=True),  # one gate per head
            nn.Sigmoid(),
        )

        # Per-head query modulation projection
        self.q_proj_delta = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.k_proj, self.v_proj, self.o_proj,
                       self.q_proj_base, self.q_proj_delta]:
            nn.init.xavier_uniform_(module.weight)
        # init gate to near-zero so model starts close to standard attention
        nn.init.zeros_(self.gate_net[-2].bias)
        nn.init.normal_(self.gate_net[0].weight, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, d_model) — query input
            mask:      (B, 1, T, T) boolean causal mask, True = keep
            key_value: (B, S, d_model) — cross-attention source (optional)
                       if None, self-attention is used

        Returns:
            out: (B, T, d_model)
        """
        B, T, _ = x.shape
        src = key_value if key_value is not None else x
        S = src.shape[1]

        # ── Standard K, V ──────────────────────────────────────────────────
        K = self.k_proj(src).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(src).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # ── Dynamic Query ───────────────────────────────────────────────────
        q_base  = self.q_proj_base(x)                          # (B, T, D)
        q_delta = self.q_proj_delta(x)                         # (B, T, D)

        # Gate: per-token, per-head scalar in [0, 1]
        gate = self.gate_net(x)                                # (B, T, n_heads)
        gate = gate.unsqueeze(-1)                              # (B, T, n_heads, 1)

        # Reshape for gating
        q_base_h  = q_base.view(B, T, self.n_heads, self.d_head)
        q_delta_h = q_delta.view(B, T, self.n_heads, self.d_head)

        # Modulated query: base + gate * delta
        Q = (q_base_h + gate * q_delta_h).transpose(1, 2)     # (B, H, T, d_head)

        # ── Scaled dot-product attention ────────────────────────────────────
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, S)

        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                            # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.resid_drop(self.o_proj(out))

        return out
