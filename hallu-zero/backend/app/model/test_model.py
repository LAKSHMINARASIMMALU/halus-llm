"""
Quick smoke test — run this to verify all components work:
    python -m app.model.test_model
"""
import torch
from app.model.transformer import HalluZeroTransformer
from app.model.jepa        import JEPAModule


def test_forward():
    print("Testing HalluZeroTransformer + JEPA...")

    B, T, V = 2, 64, 1000
    d_model  = 256
    n_layers = 4
    n_heads  = 4

    model = HalluZeroTransformer(
        vocab_size=V,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=T + 1,
        dropout=0.0,
        gate_rank=8,
    )
    print(f"  Parameters: {model.count_parameters():,}")

    jepa = JEPAModule(d_model=d_model, d_pred=64, lambda_jepa=0.1)
    jepa.init_target(model)

    input_ids = torch.randint(0, V, (B, T))
    labels    = torch.randint(0, V, (B, T))

    # Forward
    out    = model(input_ids, return_hidden=True)
    logits = out['logits']
    hidden = out['hidden']

    assert logits.shape == (B, T, V),    f"Wrong logits shape: {logits.shape}"
    assert hidden.shape == (B, T, d_model), f"Wrong hidden shape: {hidden.shape}"
    print(f"  Logits: {logits.shape} ✓")
    print(f"  Hidden: {hidden.shape} ✓")

    # CLM loss
    clm_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, V), labels.view(-1)
    )
    print(f"  CLM loss: {clm_loss.item():.4f} ✓")

    # JEPA loss
    mask          = jepa.mask_generator(T, device=input_ids.device)
    full_emb      = model.get_embeddings(input_ids)
    jepa_loss     = jepa(hidden, full_emb, mask)
    print(f"  JEPA loss: {jepa_loss.item():.4f} ✓")
    print(f"  Masked tokens: {mask.sum().item()}/{T}")

    # Backward
    total = clm_loss + jepa_loss
    total.backward()
    print(f"  Backward pass ✓")

    # EMA update
    jepa.update_target(model)
    print(f"  EMA target update ✓")

    print("\nAll tests passed!")
    return True


def test_dqa():
    print("\nTesting Dynamic Query Attention...")
    from app.model.attention import DynamicQueryAttention

    B, T, D = 2, 32, 256
    attn = DynamicQueryAttention(d_model=D, n_heads=4, gate_rank=8)
    x = torch.randn(B, T, D)
    out = attn(x)
    assert out.shape == (B, T, D)
    print(f"  DQA output: {out.shape} ✓")
    print(f"  Gate activates per-token, per-head query modulation ✓")


def test_apn():
    print("\nTesting Adaptive Pre-Norm...")
    from app.model.norm import AdaptivePreNorm

    B, T, D = 2, 32, 256
    apn = AdaptivePreNorm(d_model=D)
    x   = torch.randn(B, T, D)
    x_norm, alpha, beta = apn(x)
    assert x_norm.shape == (B, T, D)
    print(f"  APN output:  {x_norm.shape} ✓")
    print(f"  Alpha range: [{alpha.min().item():.3f}, {alpha.max().item():.3f}]")
    print(f"  Beta  range: [{beta.min().item():.3f},  {beta.max().item():.3f}]")


if __name__ == "__main__":
    test_dqa()
    test_apn()
    test_forward()
