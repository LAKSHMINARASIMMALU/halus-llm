"""
HalluZero Pretraining Script
==============================
Trains the HalluZeroTransformer with:
  1. Causal Language Modeling loss (CLM) — standard next-token prediction
  2. JEPA auxiliary loss               — abstract representation prediction

Total loss: L = L_CLM + λ * L_JEPA

Usage:
    python -m app.model.pretrain \
        --data_path ./data/pretrain_corpus.txt \
        --output_dir ./data/checkpoints \
        --d_model 512 \
        --n_layers 8 \
        --n_heads 8 \
        --batch_size 8 \
        --max_steps 10000 \
        --lr 3e-4

Requirements:
    pip install torch transformers datasets tqdm

The trained model can then be used as the base LLM instead of Ollama
for full end-to-end control of the hallucination reduction pipeline.
"""

import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from pathlib import Path

from app.model.transformer import HalluZeroTransformer
from app.model.jepa        import JEPAModule


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Simple character/token dataset from a text file."""

    def __init__(self, path: str, seq_len: int = 512, tokenizer=None):
        self.seq_len = seq_len
        text = Path(path).read_text(encoding="utf-8")

        if tokenizer is not None:
            self.tokens = tokenizer.encode(text)
        else:
            # Character-level fallback for testing
            chars = sorted(set(text))
            self.stoi = {c: i for i, c in enumerate(chars)}
            self.itos = {i: c for c, i in self.stoi.items()}
            self.vocab_size = len(chars)
            self.tokens = [self.stoi[c] for c in text]

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


# ── Trainer ───────────────────────────────────────────────────────────────────

class HalluZeroPretrainer:
    """
    Full pretraining loop with CLM + JEPA.

    Architecture connections:
      - model.encode()    → used as JEPA context encoder
      - jepa.predictor    → predicts masked representations
      - jepa.target_encoder (EMA copy) → provides stable targets
    """

    def __init__(
        self,
        model: HalluZeroTransformer,
        jepa: JEPAModule,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        output_dir: str,
        log_every: int = 100,
        save_every: int = 1000,
        grad_clip: float = 1.0,
        tau_schedule: bool = True,
    ):
        self.model       = model
        self.jepa        = jepa
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.output_dir  = Path(output_dir)
        self.log_every   = log_every
        self.save_every  = save_every
        self.grad_clip   = grad_clip
        self.tau_schedule = tau_schedule
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def _tau(self, max_steps: int) -> float:
        """Cosine schedule: tau 0.996 → 1.0 over training."""
        if not self.tau_schedule:
            return 0.996
        return 1.0 - (1.0 - 0.996) * (math.cos(math.pi * self.step / max_steps) + 1) / 2

    def train_step(self, x: torch.Tensor, y: torch.Tensor, max_steps: int) -> dict:
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        B, T = x.shape

        # ── Forward pass ────────────────────────────────────────────────────
        out    = self.model(x, return_hidden=True)
        logits = out['logits']    # (B, T, V)
        hidden = out['hidden']    # (B, T, D)

        # ── CLM loss ─────────────────────────────────────────────────────────
        clm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
        )

        # ── JEPA loss ────────────────────────────────────────────────────────
        mask = self.jepa.mask_generator(T, device=self.device)
        full_embeddings = self.model.get_embeddings(x)   # (B, T, D) — input to target enc
        jepa_loss = self.jepa(
            context_reps=hidden,
            full_embeddings=full_embeddings,
            mask=mask,
        )

        # ── Total loss ───────────────────────────────────────────────────────
        total_loss = clm_loss + jepa_loss

        # ── Backward ─────────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        # ── EMA update ───────────────────────────────────────────────────────
        tau = self._tau(max_steps)
        self.jepa.update_target(self.model, tau=tau)

        self.step += 1
        return {
            "loss":      total_loss.item(),
            "clm_loss":  clm_loss.item(),
            "jepa_loss": jepa_loss.item(),
            "lr":        self.scheduler.get_last_lr()[0],
            "tau":       tau,
            "ppl":       math.exp(min(clm_loss.item(), 10)),
        }

    def train(self, dataloader: DataLoader, max_steps: int):
        print(f"Starting pretraining — {max_steps} steps")
        print(f"Model params: {self.model.count_parameters():,}")

        losses = []
        data_iter = iter(dataloader)

        for step in range(max_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            metrics = self.train_step(x, y, max_steps)
            losses.append(metrics["loss"])

            if step % self.log_every == 0:
                avg = sum(losses[-self.log_every:]) / min(len(losses), self.log_every)
                print(
                    f"step {step:6d} | loss {metrics['loss']:.4f} | "
                    f"clm {metrics['clm_loss']:.4f} | "
                    f"jepa {metrics['jepa_loss']:.4f} | "
                    f"ppl {metrics['ppl']:.2f} | "
                    f"lr {metrics['lr']:.2e} | "
                    f"tau {metrics['tau']:.4f}"
                )

            if step % self.save_every == 0 and step > 0:
                self.save(step)

        self.save(max_steps)
        print("Pretraining complete.")

    def save(self, step: int):
        ckpt = {
            "step":        step,
            "model":       self.model.state_dict(),
            "jepa":        self.jepa.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
        }
        path = self.output_dir / f"checkpoint_{step:08d}.pt"
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HalluZero Pretrainer")
    parser.add_argument("--data_path",   default="./data/pretrain_corpus.txt")
    parser.add_argument("--output_dir",  default="./data/checkpoints")
    parser.add_argument("--d_model",     type=int, default=512)
    parser.add_argument("--n_layers",    type=int, default=8)
    parser.add_argument("--n_heads",     type=int, default=8)
    parser.add_argument("--seq_len",     type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--max_steps",   type=int, default=10000)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--warmup_steps",type=int, default=500)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--gate_rank",   type=int, default=16)
    parser.add_argument("--lambda_jepa", type=float, default=0.1)
    parser.add_argument("--vocab_size",  type=int, default=None)
    parser.add_argument("--use_hf_tokenizer", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset & tokenizer ────────────────────────────────────────────────
    tokenizer = None
    if args.use_hf_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
    else:
        vocab_size = args.vocab_size

    dataset = TextDataset(args.data_path, seq_len=args.seq_len, tokenizer=tokenizer)
    if vocab_size is None:
        vocab_size = dataset.vocab_size
    print(f"Vocab size: {vocab_size} | Dataset size: {len(dataset):,} sequences")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = HalluZeroTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len + 1,
        dropout=args.dropout,
        gate_rank=args.gate_rank,
    ).to(device)

    # ── JEPA ───────────────────────────────────────────────────────────────
    jepa = JEPAModule(
        d_model=args.d_model,
        d_pred=args.d_model // 4,
        lambda_jepa=args.lambda_jepa,
    ).to(device)
    jepa.init_target(model)

    # ── Optimizer + cosine LR schedule ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(jepa.predictor.parameters()),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Train ──────────────────────────────────────────────────────────────
    trainer = HalluZeroPretrainer(
        model=model,
        jepa=jepa,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
    )
    trainer.train(dataloader, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
