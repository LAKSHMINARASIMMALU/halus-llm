"""
JEPA — Joint Embedding Predictive Architecture
===============================================
Original: Assran et al. "Self-Supervised Learning from Images with a
          Joint-Embedding Predictive Architecture" (I-JEPA, 2023)
This implementation: language adaptation for LLM pretraining.

Core idea:
  Instead of predicting raw tokens (like standard MLM/CLM),
  JEPA predicts the *abstract representation* of masked spans
  in embedding space. This forces the encoder to learn causal,
  semantic structure rather than surface-level patterns.

Architecture:
  ┌─────────────────────────────────────────────┐
  │  Context encoder (full transformer)          │
  │  Takes: visible tokens (unmasked)            │
  │  Outputs: context representations            │
  └──────────────────┬──────────────────────────┘
                     │ context representations
  ┌──────────────────▼──────────────────────────┐
  │  Predictor (narrow transformer)              │
  │  Takes: context reps + mask position queries │
  │  Outputs: predicted representations          │
  └──────────────────┬──────────────────────────┘
                     │ predicted reps
                     ▼  L2 distance loss
  ┌─────────────────────────────────────────────┐
  │  Target encoder (EMA copy of context enc)   │
  │  Takes: FULL tokens (including masked)       │
  │  Outputs: target representations (no grad)  │
  └─────────────────────────────────────────────┘

Training signal: MSE(predicted_reps, target_reps.detach())

The target encoder is updated via exponential moving average (EMA)
— never directly trained — ensuring stable targets.

Integration with main LLM:
  The context encoder IS the main transformer encoder.
  JEPA loss is added as an auxiliary loss during pretraining:
      total_loss = CLM_loss + λ * JEPA_loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# ── Mask generator ──────────────────────────────────────────────────────────

class BlockMaskGenerator:
    """
    Generates contiguous block masks for JEPA.
    Masks 15-40% of the sequence in 1-4 contiguous blocks.
    Contiguous blocks are better than random masking for language JEPA
    because they force prediction of coherent semantic spans.
    """

    def __init__(
        self,
        mask_ratio_min: float = 0.15,
        mask_ratio_max: float = 0.40,
        n_mask_blocks: int = 2,
    ):
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.n_mask_blocks  = n_mask_blocks

    def __call__(self, seq_len: int, device: torch.device) -> torch.BoolTensor:
        """
        Returns a bool mask of shape (seq_len,)
        True = masked (target), False = visible (context)
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        ratio = self.mask_ratio_min + torch.rand(1).item() * (
            self.mask_ratio_max - self.mask_ratio_min
        )
        n_mask = max(1, int(seq_len * ratio))
        block_size = n_mask // self.n_mask_blocks

        for _ in range(self.n_mask_blocks):
            start = torch.randint(0, seq_len - block_size, (1,)).item()
            mask[start: start + block_size] = True

        return mask


# ── Predictor (narrow transformer) ──────────────────────────────────────────

class JEPAPredictor(nn.Module):
    """
    Lightweight predictor that maps context representations +
    learnable mask tokens → predicted target representations.

    Narrower than the main encoder (d_model // 4 typically).
    """

    def __init__(self, d_model: int, d_pred: int, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.input_proj  = nn.Linear(d_model, d_pred)
        self.output_proj = nn.Linear(d_pred, d_model)

        # Learnable mask token (replaces masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding for predictor
        self.pos_emb = nn.Embedding(4096, d_pred)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_pred,
            nhead=n_heads,
            dim_feedforward=d_pred * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        context_reps: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Args:
            context_reps: (B, T, d_model) — full-sequence context encoder output
            mask:         (T,) or (B, T) bool — True = masked positions to predict

        Returns:
            predicted_reps: (B, T, d_model) — predictions for ALL positions
                            (only masked positions matter for the loss)
        """
        B, T, _ = context_reps.shape
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(B, -1)  # (B, T)

        # Project context to predictor dimension
        x = self.input_proj(context_reps)             # (B, T, d_pred)

        # Replace masked positions with learned mask token
        mask_tokens = self.mask_token.expand(B, T, -1)
        x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

        # Add positional embeddings
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_emb(positions).unsqueeze(0)

        # Run predictor transformer
        x = self.transformer(x)                        # (B, T, d_pred)

        # Project back to d_model
        return self.output_proj(x)                     # (B, T, d_model)


# ── EMA target encoder ───────────────────────────────────────────────────────

class EMATargetEncoder(nn.Module):
    """
    Maintains an exponential moving average copy of the context encoder.
    The target encoder is NEVER directly optimized — it provides stable
    prediction targets, preventing representation collapse.

    EMA update: θ_target = τ * θ_target + (1-τ) * θ_online
    τ starts at 0.996 and increases toward 1.0 over training
    (following MoCo / BYOL / I-JEPA conventions).
    """

    def __init__(self, online_encoder: nn.Module, tau: float = 0.996):
        super().__init__()
        self.tau = tau
        # Deep copy of encoder — no gradient tracking
        import copy
        self.target = copy.deepcopy(online_encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module, tau: float = None):
        """Call after every optimizer step."""
        tau = tau or self.tau
        for t_param, o_param in zip(
            self.target.parameters(), online_encoder.parameters()
        ):
            t_param.data.mul_(tau).add_(o_param.data, alpha=1.0 - tau)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.target(x, **kwargs)


# ── JEPA loss ────────────────────────────────────────────────────────────────

class JEPALoss(nn.Module):
    """
    Computes the JEPA auxiliary loss:
        L_JEPA = MSE(predicted_reps[mask], target_reps[mask].detach())

    Normalized by the L2 norm of targets for stability.
    """

    def forward(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (B, T, D) — predictor output
            targets:   (B, T, D) — target encoder output (no grad)
            mask:      (B, T) or (T,) bool — True = positions to compute loss on

        Returns:
            scalar loss
        """
        if mask.dim() == 1:
            B = predicted.shape[0]
            mask = mask.unsqueeze(0).expand(B, -1)

        pred_masked   = predicted[mask]    # (N, D)
        target_masked = targets[mask]      # (N, D)

        # Normalize targets (prevents loss from collapsing to 0)
        target_masked = F.normalize(target_masked, dim=-1)
        pred_masked   = F.normalize(pred_masked,   dim=-1)

        loss = F.mse_loss(pred_masked, target_masked.detach())
        return loss


# ── Complete JEPA module ──────────────────────────────────────────────────────

class JEPAModule(nn.Module):
    """
    Full JEPA module to be used alongside the main transformer.

    Usage during pretraining:
        jepa = JEPAModule(d_model=512, d_pred=128)

        # Forward pass
        mask = jepa.mask_generator(seq_len=128, device=x.device)
        context_reps = main_encoder(x)        # your transformer output
        jepa_loss = jepa(context_reps, x_embeddings, mask)

        total_loss = clm_loss + 0.1 * jepa_loss
        total_loss.backward()
        jepa.update_target(main_encoder)       # EMA update
    """

    def __init__(
        self,
        d_model: int,
        d_pred: int = None,
        tau: float = 0.996,
        lambda_jepa: float = 0.1,
        mask_ratio_min: float = 0.15,
        mask_ratio_max: float = 0.40,
    ):
        super().__init__()
        d_pred = d_pred or max(64, d_model // 4)
        self.lambda_jepa   = lambda_jepa
        self.mask_generator = BlockMaskGenerator(mask_ratio_min, mask_ratio_max)
        self.predictor      = JEPAPredictor(d_model, d_pred)
        self.loss_fn        = JEPALoss()
        self._target: EMATargetEncoder = None  # set via init_target()

    def init_target(self, online_encoder: nn.Module):
        """Call once before training starts."""
        self.target_encoder = EMATargetEncoder(online_encoder)

    def update_target(self, online_encoder: nn.Module, tau: float = None):
        """Call after every optimizer step."""
        if self.target_encoder is not None:
            self.target_encoder.update(online_encoder, tau=tau)

    def forward(
        self,
        context_reps: torch.Tensor,
        full_embeddings: torch.Tensor,
        mask: torch.BoolTensor,
        encoder_kwargs: dict = None,
    ) -> torch.Tensor:
        """
        Args:
            context_reps:    (B, T, D) — main encoder output on visible tokens
            full_embeddings: (B, T, D) — token embeddings for target encoder
            mask:            (T,) bool — True = masked
            encoder_kwargs:  extra kwargs to pass to target encoder

        Returns:
            jepa_loss * lambda_jepa  (scalar)
        """
        # Predict masked representations from context
        predicted = self.predictor(context_reps, mask)   # (B, T, D)

        # Get target representations (EMA encoder, no grad)
        if self.target_encoder is not None:
            kw = encoder_kwargs or {}
            with torch.no_grad():
                targets = self.target_encoder(full_embeddings, **kw)
        else:
            targets = full_embeddings  # fallback during testing

        B = predicted.shape[0]
        if mask.dim() == 1:
            mask_expanded = mask.unsqueeze(0).expand(B, -1)
        else:
            mask_expanded = mask

        loss = self.loss_fn(predicted, targets, mask_expanded)
        return loss * self.lambda_jepa
