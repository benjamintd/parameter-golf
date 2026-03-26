#!/usr/bin/env python3
"""
Parameter Golf local benchmark — scaled-down training on FineWeb with
competition-style reporting, loss curves, and val_bpb evaluation.

Usage:
    uv run python run_benchmark.py                       # defaults
    uv run python run_benchmark.py envs/mps_small.env    # with env file

Produces:
  - benchmark_report/loss_curve.png
  - benchmark_report/submission.json
  - benchmark_report/README.md
"""

import torch
import torch.nn.functional as F
import math
import time
import json
import zlib
import io
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Load .env file if passed as argument
if len(sys.argv) > 1 and sys.argv[1].endswith(".env"):
    env_path = Path(sys.argv[1])
    if env_path.exists():
        print(f"Loading env: {env_path}")
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
    else:
        print(f"WARNING: {env_path} not found, using defaults")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_gpt import (
    GolfModel,
    SpeedrunMoE,
    Hyperparameters,
    TokenStream,
    load_data_shard,
    load_tokenizer_byte_tables,
    eval_val,
    quantize_state_dict,
    quantize_int6_per_row,
)

# ---------------------------------------------------------------------------
# Config — all overridable via environment variables
# ---------------------------------------------------------------------------
REPORT_DIR = Path(os.environ.get("REPORT_DIR", "benchmark_report"))

VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 1024))
DIM = int(os.environ.get("MODEL_DIM", os.environ.get("DIM", 512)))
NUM_HEADS = int(os.environ.get("NUM_HEADS", 8))
DEPTH = int(os.environ.get("DEPTH", 12))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 64))
HIDDEN_MULT = int(os.environ.get("MLP_MULT", os.environ.get("HIDDEN_MULT", 2)))
SEQ_LEN = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("SEQ_LEN", 1024)))
TOP_K = int(os.environ.get("TOP_K", 2))

LR = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
NUM_STEPS = int(os.environ.get("NUM_STEPS", os.environ.get("ITERATIONS", 2000)))
VAL_EVERY = int(os.environ.get("VAL_EVERY", 100))
VAL_TOKENS_QUICK = int(os.environ.get("VAL_TOKENS_QUICK", 200_000))
VAL_TOKENS_FULL = int(os.environ.get("VAL_TOKENS_FULL", 2_000_000))

COMPETITION_SCORES = {
    "Naive Baseline (9L 512d, 8xH100 10min)": 1.2244,
    "11L EMA+GPTQ-lite (signalrush)": 1.1228,
    "11L PartialRoPE+LN (jfprincz)": 1.1248,
    "LeakyReLU^2+TTT+Muon (abaybektursun, SOTA)": 1.1194,
}

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_batches(tokens, batch_size, seq_len):
    total_seqs = (tokens.numel() - 1) // seq_len
    usable_seqs = (total_seqs // batch_size) * batch_size
    for i in range(0, usable_seqs, batch_size):
        start = i * seq_len
        end = (i + batch_size) * seq_len + 1
        chunk = tokens[start:end]
        x = chunk[:-1].reshape(batch_size, seq_len)
        y = chunk[1:].reshape(batch_size, seq_len)
        yield x, y


def compute_artifact_size(model):
    with torch.no_grad():
        qsd = quantize_state_dict(model.state_dict())
        buf = io.BytesIO()
        torch.save(qsd, buf)
        raw = buf.getvalue()
        compressed = zlib.compress(raw, level=9)
    return len(raw), len(compressed)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_loss_curve(log, report_dir):
    steps = [e["step"] for e in log]
    losses = [e["train_loss"] for e in log]
    val_steps = [e["step"] for e in log if "val_bpb" in e]
    val_bpbs = [e["val_bpb"] for e in log if "val_bpb" in e]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, losses, linewidth=1.2, color="#2563eb", label="Train loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss (nats)")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if val_bpbs:
        ax2.plot(val_steps, val_bpbs, "o-", linewidth=1.5, color="#2563eb",
                 markersize=5, label="This run (local approx)")
        colors = ["#dc2626", "#ea580c", "#d97706", "#16a34a"]
        for (name, score), color in zip(COMPETITION_SCORES.items(), colors):
            ax2.axhline(y=score, linestyle="--", linewidth=1, color=color,
                        alpha=0.7, label=f"{name}: {score}")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("val_bpb (bits/byte)")
        ax2.set_title("Validation BPB")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7, loc="upper right")
        min_bpb = min(val_bpbs)
        ax2.set_ylim(bottom=max(0.9, min(1.1, min_bpb - 0.1)))

    plt.tight_layout()
    path = report_dir / "loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(log, model, elapsed, total_tokens_seen, report_dir):
    final = [e for e in log if "val_bpb" in e][-1]
    val_bpb = final["val_bpb"]
    val_loss = final["val_loss"]
    total_params = sum(p.numel() for p in model.parameters())
    raw_bytes, compressed_bytes = compute_artifact_size(model)
    code_bytes = len(Path("train_gpt.py").read_text().encode("utf-8")) if Path("train_gpt.py").exists() else 0
    total_submission = compressed_bytes + code_bytes

    submission = {
        "author": "local-benchmark",
        "github_id": "",
        "name": f"GolfModel {DIM}d {DEPTH}L {NUM_EXPERTS}E MoE+Passthrough",
        "blurb": (
            f"Depth-recurrent transformer with MoE (passthrough expert 0), "
            f"timestep embedding, RoPE. {DIM}d, {DEPTH}L, {NUM_EXPERTS}E, top-{TOP_K}."
        ),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "val_loss": round(val_loss, 8),
        "val_bpb": round(val_bpb, 7),
        "bytes_total": total_submission,
        "bytes_code": code_bytes,
        "meta": {
            "device": str(DEVICE),
            "total_params": total_params,
            "training_steps": NUM_STEPS,
            "tokens_seen": total_tokens_seen,
            "wall_time_seconds": round(elapsed, 1),
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
        },
    }

    sub_path = report_dir / "submission.json"
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"  Saved: {sub_path}")

    tokens_per_sec = total_tokens_seen / elapsed if elapsed > 0 else 0
    readme = f"""# Benchmark Report

## Run Info

| Field | Value |
|-------|-------|
| Date | {submission['date']} |
| Device | {DEVICE} |
| Wall time | {elapsed:.1f}s |
| Steps | {NUM_STEPS} |
| Tokens seen | {total_tokens_seen:,} |
| Tokens/sec | {tokens_per_sec:,.0f} |

## Model

| Field | Value |
|-------|-------|
| Architecture | GolfModel (depth-recurrent transformer + MoE) |
| dim | {DIM} |
| num_heads | {NUM_HEADS} |
| depth (recurrence) | {DEPTH} |
| MoE experts | {NUM_EXPERTS} (1 passthrough + {NUM_EXPERTS-1} real, top-{TOP_K}) |
| hidden_dim | {DIM * HIDDEN_MULT} |
| vocab_size | {VOCAB_SIZE} |
| seq_len | {SEQ_LEN} |
| Total params | {total_params:,} |
| Tied embeddings | Yes |
| Positional encoding | RoPE + timestep embedding |

## Results

| Metric | Value |
|--------|-------|
| **val_bpb** | **{val_bpb:.4f}** |
| val_loss (nats) | {val_loss:.4f} |
| Artifact (raw) | {raw_bytes / 1e6:.2f} MB |
| Artifact (zlib) | {compressed_bytes / 1e6:.2f} MB |
| Code size | {code_bytes / 1e3:.1f} KB |
| Total submission | {total_submission / 1e6:.2f} MB |
| Under 16MB? | {'Yes' if total_submission <= 16_000_000 else 'No'} |

## Comparison to Competition (8xH100 10min)

| Submission | val_bpb |
|-----------|---------|
"""
    for name, score in COMPETITION_SCORES.items():
        readme += f"| {name} | {score:.4f} |\n"
    readme += f"| **This run (local approx)** | **{val_bpb:.4f}** |\n"
    readme += f"""
## Loss Curve

![Loss Curve](loss_curve.png)

## Training Log (sampled)

| Step | Train Loss | val_bpb | Tokens |
|------|-----------|---------|--------|
"""
    for e in log:
        bpb_str = f"{e['val_bpb']:.4f}" if "val_bpb" in e else ""
        readme += f"| {e['step']} | {e['train_loss']:.4f} | {bpb_str} | {e['tokens_seen']:,} |\n"

    readme_path = report_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"  Saved: {readme_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = Hyperparameters()
    DATA_DIR = Path(args.data_path)
    train_shard = sorted(Path(args.data_path).glob("fineweb_train_*.bin"))
    val_shard = sorted(Path(args.data_path).glob("fineweb_val_*.bin"))

    if not train_shard or not val_shard:
        # Fallback to local data dir
        DATA_DIR = Path("data/fineweb")
        train_shard = sorted(DATA_DIR.glob("fineweb_train_*.bin"))
        val_shard = sorted(DATA_DIR.glob("fineweb_val_*.bin"))

    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        tokenizer_path = Path("data/tokenizers/fineweb_1024_bpe.model")

    for p in [*train_shard[:1], *val_shard[:1], tokenizer_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Download data first.")
            return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading data...")
    train_tokens = torch.cat([load_data_shard(f) for f in train_shard])
    val_tokens = torch.cat([load_data_shard(f) for f in val_shard])
    print(f"  Train: {train_tokens.numel():,} tokens")
    print(f"  Val:   {val_tokens.numel():,} tokens")

    print(f"Loading tokenizer...")
    base_bytes, has_leading_space, is_boundary = load_tokenizer_byte_tables(str(tokenizer_path), VOCAB_SIZE)

    print(f"Building model...")
    model = GolfModel(
        vocab_size=VOCAB_SIZE, dim=DIM, num_heads=NUM_HEADS, depth=DEPTH,
        num_experts=NUM_EXPERTS, mlp_mult=HIDDEN_MULT, top_k=TOP_K, max_seq_len=SEQ_LEN,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS)

    # Build eval args for eval_val
    eval_args = Hyperparameters()
    eval_args.val_batch_size = BATCH_SIZE * SEQ_LEN
    eval_args.train_seq_len = SEQ_LEN

    # --- Training loop ---
    print(f"\nTraining for {NUM_STEPS} steps (batch={BATCH_SIZE}, seq={SEQ_LEN})...")
    print(f"{'='*60}")

    log = []
    total_tokens_seen = 0
    batch_iter = iter(make_batches(train_tokens, BATCH_SIZE, SEQ_LEN))
    t_start = time.time()

    for step in range(1, NUM_STEPS + 1):
        try:
            x, y = next(batch_iter)
        except StopIteration:
            batch_iter = iter(make_batches(train_tokens, BATCH_SIZE, SEQ_LEN))
            x, y = next(batch_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss = loss.item()
        total_tokens_seen += y.numel()

        do_val = (step % VAL_EVERY == 0) or (step == 1) or (step == NUM_STEPS)
        entry = {
            "step": step,
            "train_loss": train_loss,
            "tokens_seen": total_tokens_seen,
            "wall_time": time.time() - t_start,
            "lr": scheduler.get_last_lr()[0],
        }

        if do_val:
            val_loss, val_bpb = eval_val(
                eval_args, model, DEVICE,
                val_tokens[:VAL_TOKENS_QUICK], base_bytes, has_leading_space, is_boundary,
            )
            entry["val_bpb"] = val_bpb
            entry["val_loss"] = val_loss
            elapsed = time.time() - t_start
            tok_s = total_tokens_seen / elapsed
            print(
                f"  step {step:4d}/{NUM_STEPS} | "
                f"loss {train_loss:.4f} | "
                f"val_bpb {val_bpb:.4f} | "
                f"lr {entry['lr']:.2e} | "
                f"{tok_s:,.0f} tok/s | "
                f"{elapsed:.0f}s"
            )
        elif step % 10 == 0:
            print(f"  step {step:4d}/{NUM_STEPS} | loss {train_loss:.4f}", end="\r")

        log.append(entry)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s")

    # --- Final evaluation ---
    print(f"\nFinal evaluation ({VAL_TOKENS_FULL:,} val tokens)...")
    final_loss, final_bpb = eval_val(
        eval_args, model, DEVICE,
        val_tokens[:VAL_TOKENS_FULL], base_bytes, has_leading_space, is_boundary,
    )
    print(f"  val_loss = {final_loss:.4f} nats")
    print(f"  val_bpb  = {final_bpb:.4f} bits/byte")

    log[-1]["val_bpb"] = final_bpb
    log[-1]["val_loss"] = final_loss

    # --- Generate outputs ---
    print(f"\nGenerating report...")
    plot_loss_curve(log, REPORT_DIR)
    generate_report(log, model, elapsed, total_tokens_seen, REPORT_DIR)

    log_path = REPORT_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: {log_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  val_bpb:  {final_bpb:.4f}")
    print(f"  val_loss: {final_loss:.4f} nats")
    raw_bytes, comp_bytes = compute_artifact_size(model)
    print(f"  Artifact: {comp_bytes / 1e6:.2f} MB (zlib), {'OK' if comp_bytes <= 16_000_000 else 'OVER'}")
    print()
    print(f"  Competition reference:")
    for name, score in COMPETITION_SCORES.items():
        delta = final_bpb - score
        print(f"    {score:.4f}  {name}  (delta: +{delta:.4f})")
    print()
    print(f"  Report: {REPORT_DIR}/")


if __name__ == "__main__":
    main()
