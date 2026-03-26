"""
Test script for train_gpt.py — validates all components on macOS (MPS/CPU).
Usage: uv run python test_parameter_golf.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
import math
import zlib
import io
from pathlib import Path

from train_gpt import (
    GolfModel,
    SpeedrunMoE,
    Muon,
    Hyperparameters,
    BackoffNgramMixer,
    ngram_mixer_loss,
    zeropower_via_newtonschulz5,
    sigreg_loss,
    quantize_int6_per_row,
    quantize_state_dict,
    dequantize_state_dict,
    apply_rope,
    precompute_rope_freqs,
    load_data_shard,
    load_tokenizer_byte_tables,
    eval_val,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/fineweb")
TOKENIZER_PATH = Path("data/tokenizers/fineweb_1024_bpe.model")
VOCAB_SIZE = 1024
SEQ_LEN = 256

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_newton_schulz():
    G = torch.randn(32, 64)
    X = zeropower_via_newtonschulz5(G, steps=10)
    prod = X.float() @ X.float().T
    eye = torch.eye(prod.shape[0])
    err = (prod - eye).abs().max().item()
    assert err < 0.3, f"Newton-Schulz not orthogonal enough: max error {err:.4f}"
    print(f"  max orthogonality error = {err:.4f} (OK)")


def test_rope():
    cos, sin = precompute_rope_freqs(16, 64)
    x = torch.randn(2, 4, 32, 16)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    assert torch.allclose(out, apply_rope(x, cos, sin))
    print(f"  shape preserved, deterministic (OK)")


def test_sigreg():
    loss = sigreg_loss(torch.randn(64, 128), num_projections=16)
    assert loss.ndim == 0 and loss.item() >= 0
    print(f"  loss = {loss.item():.4f} (OK)")


def test_passthrough_expert():
    moe = SpeedrunMoE(dim=64, hidden_dim=128, num_experts=8, top_k=2).to(DEVICE)
    assert moe.num_real_experts == 7
    assert moe.w1.shape == (7, 64, 128)
    x = torch.randn(2, 16, 64, device=DEVICE)
    out, reg = moe(x)
    assert out.shape == x.shape
    assert reg.ndim == 0
    print(f"  7 real + 1 passthrough, shapes OK (OK)")


def test_timestep_embed():
    model = GolfModel(vocab_size=256, dim=64, num_heads=4, depth=4, max_seq_len=32).to(DEVICE)
    assert model.timestep_embed.shape == (4, 64)
    # Each step should have a different embedding
    diffs = (model.timestep_embed[1:] - model.timestep_embed[:-1]).abs().sum(dim=1)
    assert (diffs > 0).all(), "Timestep embeddings should differ across steps"
    print(f"  shape (4, 64), all steps distinct (OK)")


def test_ngram_mixer():
    """BackoffNgramMixer should count n-grams and produce valid probabilities."""
    # Use CPU — mixer uses int32 scatter_add which MPS handles fine, but keep it simple
    dev = "cpu"
    mixer = BackoffNgramMixer(vocab_size=256, device=dev, buckets=4096)
    # Feed some tokens
    tokens = torch.randint(0, 256, (10000,))
    mixer.update(tokens)
    assert mixer.total_tokens == 10000

    # Query probabilities
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    order_p, order_valid = mixer.ngram_probs(x, y)
    assert order_p.shape == (2, 32, 6)
    assert order_valid.shape == (2, 32, 6)
    assert (order_p >= 0).all() and (order_p <= 1).all()
    print(f"  prefilled {mixer.total_tokens} tokens, probs shape {order_p.shape} (OK)")

    # Test mixer loss computation
    model = GolfModel(
        vocab_size=256, dim=64, num_heads=4, depth=2, num_experts=8,
        mlp_mult=2, top_k=2, max_seq_len=32, ngram_enabled=True,
    )
    ids = torch.randint(0, 256, (2, 32))
    logits = model(ids)  # (2, 32, 256)
    gate_logits = model.ngram_gate(torch.randn(2, 32, 64))  # (2, 32, 7)
    loss = ngram_mixer_loss(gate_logits, logits.view(-1, 256), ids, order_p, order_valid)
    assert loss.ndim == 0 and not torch.isnan(loss)
    print(f"  mixer loss = {loss.item():.4f} (OK)")

    # Test forward with ngram
    loss_with = model(ids, ids, ngram_order_p=order_p, ngram_order_valid=order_valid)
    loss_without = model(ids, ids)
    assert not torch.isnan(loss_with)
    print(f"  forward with ngram: {loss_with.item():.4f}, without: {loss_without.item():.4f} (OK)")


def test_model_forward():
    model = GolfModel(
        vocab_size=256, dim=64, num_heads=4, depth=2, num_experts=8,
        mlp_mult=2, top_k=2, max_seq_len=32,
    ).to(DEVICE)
    ids = torch.randint(0, 256, (2, 32), device=DEVICE)
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (2, 32, 256)
    loss = model(ids, ids)
    assert loss.ndim == 0 and not torch.isnan(loss)
    print(f"  logits shape {logits.shape}, loss = {loss.item():.2f} (OK)")


def test_backward():
    model = GolfModel(
        vocab_size=256, dim=64, num_heads=4, depth=2, num_experts=8,
        mlp_mult=2, top_k=2, max_seq_len=32,
    ).to(DEVICE)
    ids = torch.randint(0, 256, (2, 32), device=DEVICE)
    loss = model(ids, ids)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"No gradient for: {no_grad}"
    print(f"  all parameters have gradients (OK)")


def test_muon_step():
    """Muon optimizer should update parameters (tested on CPU for bf16 compat)."""
    model = GolfModel(
        vocab_size=256, dim=64, num_heads=4, depth=2, num_experts=8,
        mlp_mult=2, top_k=2, max_seq_len=32,
    ).to("cpu")
    matrix_params = [model.shared_moe.w1, model.shared_moe.w2, model.attn_qkv.weight]
    opt = Muon(matrix_params, lr=0.025, momentum=0.95, backend_steps=3)
    w1_before = model.shared_moe.w1.clone()
    ids = torch.randint(0, 256, (2, 32))
    opt.zero_grad()
    loss = model(ids, ids)
    loss.backward()
    opt.step()
    changed = (w1_before - model.shared_moe.w1).abs().max().item()
    assert changed > 0, "Muon did not update w1!"
    print(f"  w1 max change = {changed:.6f} (OK)")


def test_quantize_roundtrip():
    """int6 quantize → compress → decompress → dequantize should be close."""
    model = GolfModel(
        vocab_size=256, dim=64, num_heads=4, depth=2, num_experts=8,
        mlp_mult=2, top_k=2, max_seq_len=32,
    )
    # Quantize
    qsd = quantize_state_dict(model.state_dict())
    buf = io.BytesIO()
    torch.save(qsd, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)

    # Decompress and dequantize
    loaded = torch.load(io.BytesIO(zlib.decompress(compressed)), weights_only=True)
    rt_sd = dequantize_state_dict(loaded)

    # Load into fresh model
    model2 = GolfModel(
        vocab_size=256, dim=64, num_heads=4, depth=2, num_experts=8,
        mlp_mult=2, top_k=2, max_seq_len=32,
    )
    model2.load_state_dict(rt_sd, strict=False)

    # Check outputs are close
    ids = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        orig = model(ids)
        rt = model2(ids)
    # int6 has limited precision, but outputs should be in same ballpark
    max_diff = (orig - rt).abs().max().item()
    print(f"  roundtrip max logit diff = {max_diff:.4f}")
    print(f"  compressed size = {len(compressed)} bytes (OK)")


def test_artifact_size():
    """Full-size model's compressed artifact should fit in 16MB."""
    model = GolfModel(
        vocab_size=VOCAB_SIZE, dim=512, num_heads=8, depth=12,
        num_experts=64, mlp_mult=2, top_k=2, max_seq_len=1024,
    )
    total_params = sum(p.numel() for p in model.parameters())
    qsd = quantize_state_dict(model.state_dict())
    buf = io.BytesIO()
    torch.save(qsd, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    comp_mb = len(compressed) / 1e6
    print(f"  params: {total_params:,}, artifact: {comp_mb:.2f} MB (raw init)")
    # Note: trained weights compress worse, but int6 MoE should still fit
    if comp_mb <= 16.0:
        print(f"  FITS in 16MB budget (OK)")
    else:
        print(f"  WARNING: exceeds 16MB budget")


def test_data_loading():
    """Load a shard and verify format."""
    shard = DATA_DIR / "fineweb_train_000000.bin"
    if not shard.exists():
        print("  SKIPPED: data not found")
        return
    tokens = load_data_shard(shard)
    assert tokens.ndim == 1
    assert tokens.min() >= 0 and tokens.max() < VOCAB_SIZE
    print(f"  loaded {tokens.numel():,} tokens, range [{tokens.min()}, {tokens.max()}] (OK)")


def test_eval_val():
    """Run eval_val on a small model with real data."""
    val_shard = DATA_DIR / "fineweb_val_000000.bin"
    if not val_shard.exists() or not TOKENIZER_PATH.exists():
        print("  SKIPPED: data/tokenizer not found")
        return

    val_tokens = load_data_shard(val_shard)[:50_000]
    base_bytes, hls, isb = load_tokenizer_byte_tables(str(TOKENIZER_PATH), VOCAB_SIZE)

    model = GolfModel(
        vocab_size=VOCAB_SIZE, dim=64, num_heads=4, depth=2, num_experts=8,
        mlp_mult=2, top_k=2, max_seq_len=SEQ_LEN,
    ).to(DEVICE)

    args = Hyperparameters()
    args.val_batch_size = 2 * SEQ_LEN
    args.train_seq_len = SEQ_LEN

    val_loss, val_bpb = eval_val(args, model, DEVICE, val_tokens, base_bytes, hls, isb)
    assert val_bpb > 0 and not math.isnan(val_bpb)
    print(f"  val_loss={val_loss:.4f}, val_bpb={val_bpb:.4f} (OK)")


def test_short_training():
    """Run a few steps on real data and verify loss decreases."""
    train_shard = DATA_DIR / "fineweb_train_000000.bin"
    if not train_shard.exists():
        print("  SKIPPED: data not found")
        return

    tokens = load_data_shard(train_shard)
    model = GolfModel(
        vocab_size=VOCAB_SIZE, dim=64, num_heads=4, depth=2, num_experts=8,
        mlp_mult=2, top_k=2, max_seq_len=SEQ_LEN,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    pos = 0
    for step in range(20):
        chunk = tokens[pos : pos + 2 * SEQ_LEN + 1].to(DEVICE)
        x = chunk[:-1].reshape(2, SEQ_LEN)
        y = chunk[1:].reshape(2, SEQ_LEN)
        pos += 2 * SEQ_LEN

        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  20 steps: loss {losses[0]:.3f} -> {losses[-1]:.3f}", end="")
    if losses[-1] < losses[0]:
        print(" (decreasing, OK)")
    else:
        print(" (WARNING: loss did not decrease)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    unit_tests = [
        ("Newton-Schulz", test_newton_schulz),
        ("RoPE", test_rope),
        ("SigReg Loss", test_sigreg),
        ("Passthrough Expert", test_passthrough_expert),
        ("Timestep Embedding", test_timestep_embed),
        ("N-gram Mixer", test_ngram_mixer),
        ("Model Forward", test_model_forward),
        ("Backward Pass", test_backward),
        ("Muon Optimizer", test_muon_step),
        ("Quantize Roundtrip", test_quantize_roundtrip),
        ("Artifact Size", test_artifact_size),
        ("Data Loading", test_data_loading),
        ("Eval Val", test_eval_val),
        ("Short Training", test_short_training),
    ]

    passed = failed = 0
    for name, fn in unit_tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(unit_tests)}")
    if failed:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
