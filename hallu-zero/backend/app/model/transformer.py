"""
HalluZero Transformer Block
============================
Combines:
  1. Dynamic Query Attention  (DQA)     — input-conditioned query projection
  2. Adaptive Pre-Norm        (APN)     — per-token learned normalization scale
  3. Standard FFN with SwiGLU          — gated feedforward (like LLaMA/Mistral)
  4. JEPA context conditioning          — optional, injects JEPA encoder signal
                                          into the APN conditioning vector

Block structure (pre-norm):
  x → APN(cond) → DQA  → + x  →  APN(cond) → FFN → + x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from app.model.attention import DynamicQueryAttention
from app.model.norm      import AdaptivePreNorm


# ── SwiGLU FFN ───────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    Gated feedforward network using SiLU activation.
    Used in LLaMA, Mistral, PaLM — better than standard ReLU FFN.

    out = (W1(x) * SiLU(W_gate(x))) @ W2
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)  # LLaMA convention
        # Round to multiple of 256 for efficiency
        d_ff = (d_ff + 255) // 256 * 256

        self.w1   = nn.Linear(d_model, d_ff, bias=False)
        self.wg   = nn.Linear(d_model, d_ff, bias=False)
        self.w2   = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.wg(x)) * self.w1(x)))


# ── Transformer block ─────────────────────────────────────────────────────────

class HalluZeroBlock(nn.Module):
    """
    Single transformer layer with DQA + APN + SwiGLU FFN.

    Args:
        d_model:    model dimension
        n_heads:    number of attention heads
        d_ff:       feedforward dimension (default: 8/3 * d_model)
        dropout:    dropout probability
        gate_rank:  rank for DQA gate network
        cond_dim:   JEPA conditioning dimension (None = self-conditioned)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        gate_rank: int = 16,
        cond_dim: int = None,
    ):
        super().__init__()

        # Attention sub-layer
        self.attn = DynamicQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            gate_rank=gate_rank,
        )

        # FFN sub-layer
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # Adaptive Pre-Norms (one per sub-layer)
        self.attn_norm = AdaptivePreNorm(d_model, cond_dim=cond_dim)
        self.ffn_norm  = AdaptivePreNorm(d_model, cond_dim=cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        key_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, d_model)
            mask:      (B, 1, T, T) causal mask
            cond:      (B, T, cond_dim) JEPA context signal (optional)
            key_value: (B, S, d_model) for cross-attention (optional)

        Returns:
            (B, T, d_model)
        """
        # ── Attention with APN ──────────────────────────────────────────────
        x_norm, _, _ = self.attn_norm(x, cond=cond)
        x = x + self.attn(x_norm, mask=mask, key_value=key_value)

        # ── FFN with APN ────────────────────────────────────────────────────
        x_norm, _, _ = self.ffn_norm(x, cond=cond)
        x = x + self.ffn(x_norm)

        return x


# ── Full transformer stack ───────────────────────────────────────────────────

class HalluZeroTransformer(nn.Module):
    """
    Complete decoder-only transformer with:
      - Token + learned positional embeddings
      - N x HalluZeroBlock (DQA + APN + SwiGLU)
      - Final LayerNorm + LM head
      - Optional JEPA context injection at every layer

    Args:
        vocab_size:  vocabulary size
        d_model:     model dimension
        n_layers:    number of transformer blocks
        n_heads:     attention heads
        max_seq_len: maximum sequence length
        dropout:     dropout probability
        gate_rank:   DQA gate rank
        cond_dim:    JEPA conditioning dim (None = self-conditioned APN)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        gate_rank: int = 16,
        cond_dim: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            HalluZeroBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                gate_rank=gate_rank,
                cond_dim=cond_dim,
            )
            for _ in range(n_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (tok_emb and lm_head share weights — standard practice)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.q_proj_base.weight)
            nn.init.xavier_uniform_(block.ffn.w1.weight)
            nn.init.xavier_uniform_(block.ffn.w2.weight)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask — prevents attending to future tokens."""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        mask = torch.tril(mask)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns raw embeddings — used by JEPA target encoder."""
        B, T = input_ids.shape
        tok = self.tok_emb(input_ids)
        pos = self.pos_emb(torch.arange(T, device=input_ids.device))
        return self.emb_drop(tok + pos)

    def encode(
        self,
        input_ids: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the full encoder stack — returns hidden states.
        Used by JEPA context encoder.
        """
        B, T = input_ids.shape
        x = self.get_embeddings(input_ids)
        mask = self._causal_mask(T, input_ids.device)
        for block in self.blocks:
            x = block(x, mask=mask, cond=cond)
        return self.final_norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> dict:
        """
        Args:
            input_ids:     (B, T) token ids
            cond:          (B, T, cond_dim) optional JEPA context signal
            return_hidden: if True, also return hidden states

        Returns:
            dict with:
              'logits':  (B, T, vocab_size)
              'hidden':  (B, T, d_model) if return_hidden=True
        """
        hidden = self.encode(input_ids, cond=cond)
        logits = self.lm_head(hidden)

        out = {'logits': logits}
        if return_hidden:
            out['hidden'] = hidden
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
