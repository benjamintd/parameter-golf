#!/usr/bin/env python3
"""
Parameter Golf submission — single-file train_gpt.py.

Usage (single GPU):
    torchrun --standalone --nproc_per_node=1 train_gpt.py

Usage (8xH100):
    torchrun --standalone --nproc_per_node=8 train_gpt.py

All configuration via environment variables (see Hyperparameters class).
"""
from __future__ import annotations

import glob
import io
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


# =============================================================================
# Hyperparameters — all overridable via environment variables
# =============================================================================
class Hyperparameters:
    # Paths
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", "golf_moe")

    # Training
    seed = int(os.environ.get("SEED", 1337))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # Validation
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 200))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))

    # Model
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    depth = int(os.environ.get("DEPTH", 2))
    num_experts = int(os.environ.get("NUM_EXPERTS", 3))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    expert_depth = int(os.environ.get("EXPERT_DEPTH", 2))  # layers per expert
    top_k = int(os.environ.get("TOP_K", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    # N-gram mixer
    ngram_enabled = bool(int(os.environ.get("NGRAM_ENABLED", "1")))
    ngram_mixer_loss_weight = float(os.environ.get("NGRAM_MIXER_LOSS_WEIGHT", 0.5))
    ngram_buckets = int(os.environ.get("NGRAM_BUCKETS", 1_048_576))

    # Optimizer
    muon_lr = float(os.environ.get("MUON_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    adam_lr = float(os.environ.get("ADAM_LR", 0.003))


# =============================================================================
# Data loading — binary shard format matching competition
# =============================================================================
def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)
    return torch.from_numpy(tokens_np.astype(np.int64))


class TokenStream:
    """Streams tokens from binary shards, cycling through files."""

    def __init__(self, pattern: str):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(Path(self.files[0]))
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = load_data_shard(Path(self.files[self.file_idx]))
                self.pos = 0
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


# =============================================================================
# Newton-Schulz orthogonalization
# =============================================================================
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


# =============================================================================
# Muon optimizer
# =============================================================================
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95, backend_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.bfloat16()
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum)
                update = zeropower_via_newtonschulz5(update, steps=group["backend_steps"])
                scale = max(1, p.shape[-2] / p.shape[-1]) ** 0.5
                p.add_(update.to(dtype=p.dtype), alpha=-lr * scale)


# =============================================================================
# RoPE
# =============================================================================
def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# =============================================================================
# Timestep embedding for depth recurrence
# =============================================================================
def precompute_timestep_embed(depth: int, dim: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    steps = torch.arange(depth).float()
    angles = torch.outer(steps, freqs)
    embed = torch.zeros(depth, dim)
    embed[:, 0::2] = angles.sin()
    embed[:, 1::2] = angles.cos()
    return embed


# =============================================================================
# SIGReg loss
# =============================================================================
def sigreg_loss(x: Tensor, num_projections: int = 32) -> Tensor:
    N, D = x.shape
    projections = torch.randn(D, num_projections, device=x.device, dtype=x.dtype).detach()
    projections = F.normalize(projections, dim=0)
    sketches = x @ projections
    return sketches.mean(dim=0).pow(2).mean() + ((sketches.var(dim=0) - 1.0) ** 2).mean()


# =============================================================================
# SpeedrunMoE — with passthrough expert 0
# =============================================================================
class SpeedrunMoE(nn.Module):
    """
    Mixture of Experts with a parameter-free passthrough expert (index 0).
    Tokens routed to expert 0 skip all matmuls (identity). This acts as
    learned adaptive halting for depth recurrence.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 64, top_k: int = 2,
                 expert_depth: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.num_real_experts = num_experts - 1
        self.top_k = top_k
        self.expert_depth = expert_depth
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Each expert is a multi-layer MLP: dim -> hidden -> hidden -> ... -> dim
        # Layer 0: dim -> hidden_dim (up-projection)
        # Layers 1..depth-2: hidden_dim -> hidden_dim (intermediate)
        # Layer -1: hidden_dim -> dim (down-projection)
        self.w_up = nn.Parameter(torch.empty(self.num_real_experts, dim, hidden_dim))
        self.w_down = nn.Parameter(torch.empty(self.num_real_experts, hidden_dim, dim))
        nn.init.orthogonal_(self.w_up)
        nn.init.orthogonal_(self.w_down)

        # Intermediate layers (if expert_depth > 1)
        if expert_depth > 1:
            self.w_mid = nn.ParameterList([
                nn.Parameter(torch.empty(self.num_real_experts, hidden_dim, hidden_dim))
                for _ in range(expert_depth - 1)
            ])
            for w in self.w_mid:
                nn.init.orthogonal_(w)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.shape[0]

        latent_reg_loss = sigreg_loss(x_flat)

        routing_weights = F.softmax(self.router(x_flat), dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Loop per expert — with 7 real experts each gets ~N/7 tokens,
        # large enough matmuls to saturate H100, no gather OOM
        out = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_indices = selected_experts[:, k]
            weights = routing_weights[:, k].unsqueeze(-1)

            # Expert 0 = passthrough (identity, zero compute)
            pass_mask = expert_indices == 0
            if pass_mask.any():
                out[pass_mask] += x_flat[pass_mask] * weights[pass_mask]

            # Real experts (deep MLP: up -> [mid ->]* down)
            for i in range(self.num_real_experts):
                mask = expert_indices == (i + 1)
                if not mask.any():
                    continue
                h = F.relu(x_flat[mask] @ self.w_up[i])
                if self.expert_depth > 1:
                    for w in self.w_mid:
                        h = F.relu(h @ w[i])
                out[mask] += (h @ self.w_down[i]) * weights[mask]

        return out.view(B, T, D), latent_reg_loss


# =============================================================================
# GolfModel — depth-recurrent transformer with MoE
# =============================================================================
class GolfModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 512,
        num_heads: int = 8,
        depth: int = 2,
        num_experts: int = 8,
        mlp_mult: int = 4,
        top_k: int = 2,
        expert_depth: int = 3,
        max_seq_len: int = 1024,
        tie_embeddings: bool = True,
        rope_base: float = 10000.0,
        ngram_enabled: bool = False,
        ngram_mixer_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.tie_embeddings = tie_embeddings
        self.ngram_mixer_loss_weight = ngram_mixer_loss_weight

        self.embed = nn.Embedding(vocab_size, dim)
        self.attn_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_proj = nn.Linear(dim, dim, bias=False)
        self.shared_moe = SpeedrunMoE(dim, hidden_dim=dim * mlp_mult, num_experts=num_experts, top_k=top_k, expert_depth=expert_depth)
        self.norm = nn.RMSNorm(dim)

        if not tie_embeddings:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # N-gram gate head: 7 experts (1 neural + 6 n-gram orders 2-7)
        if ngram_enabled:
            self.ngram_gate = nn.Linear(dim, 7, bias=True)
            nn.init.zeros_(self.ngram_gate.weight)
            nn.init.zeros_(self.ngram_gate.bias)
            with torch.no_grad():
                self.ngram_gate.bias[0] = 2.0  # bias toward neural at init
        else:
            self.ngram_gate = None

        rope_cos, rope_sin = precompute_rope_freqs(self.head_dim, max_seq_len, theta=rope_base)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)
        self.register_buffer("timestep_embed", precompute_timestep_embed(depth, dim))

    def _recurrence_step(self, x: Tensor, step_embed: Tensor) -> tuple[Tensor, Tensor]:
        B, T, C = x.shape
        x = x + step_embed
        x_norm = self.norm(x)

        qkv = self.attn_qkv(x_norm).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        attn_out = self.attn_proj(attn_out)
        x = x + attn_out

        moe_out, reg_loss = self.shared_moe(self.norm(x))
        x = x + moe_out
        return x, reg_loss

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        ngram_order_p: Tensor | None = None,
        ngram_order_valid: Tensor | None = None,
    ) -> Tensor:
        x = self.embed(input_ids)
        total_sigreg_loss = 0.0

        for t in range(self.depth):
            step_embed = self.timestep_embed[t]
            if self.training:
                x, reg_loss = checkpoint(self._recurrence_step, x, step_embed, use_reentrant=False)
            else:
                x, reg_loss = self._recurrence_step(x, step_embed)
            total_sigreg_loss += reg_loss

        # x is the final hidden state (B, T, dim) — used for both logits and gate
        if self.tie_embeddings:
            logits = F.linear(x, self.embed.weight)
        else:
            logits = self.lm_head(x)

        if target_ids is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss = ce_loss + (0.1 * total_sigreg_loss)

            # N-gram mixer loss
            if self.ngram_gate is not None and ngram_order_p is not None:
                gate_logits = self.ngram_gate(x)  # (B, T, 7)
                mix_loss = ngram_mixer_loss(
                    gate_logits, logits.view(-1, logits.size(-1)),
                    target_ids, ngram_order_p, ngram_order_valid,
                )
                total_loss = total_loss + self.ngram_mixer_loss_weight * mix_loss

            return total_loss
        return logits

    def forward_with_gate(self, input_ids: Tensor) -> tuple[Tensor, Tensor | None]:
        """Forward pass returning logits and gate logits (for eval with mixer)."""
        x = self.embed(input_ids)
        for t in range(self.depth):
            step_embed = self.timestep_embed[t]
            x, _ = self._recurrence_step(x, step_embed)

        if self.tie_embeddings:
            logits = F.linear(x, self.embed.weight)
        else:
            logits = self.lm_head(x)

        gate_logits = self.ngram_gate(x) if self.ngram_gate is not None else None
        return logits, gate_logits


# =============================================================================
# BackoffNgramMixer — frozen n-gram oracle with learned gate
# =============================================================================
class BackoffNgramMixer:
    """Multi-order n-gram backoff oracle (orders 2-7). GPU-native hash tables.

    Prefilled once from training data, then frozen. Provides per-token
    n-gram probabilities that a learned gate head mixes with neural predictions.
    """

    def __init__(self, vocab_size: int = 1024, device: str = "cuda",
                 buckets: int = 1_048_576):
        self.V = vocab_size
        self.device = torch.device(device)
        self.total_tokens = 0
        self.max_order = 7
        self.min_order = 2
        self.BUCKETS = buckets
        self.primes = torch.tensor(
            [36313, 27191, 51647, 81929, 131071, 174763, 233017],
            dtype=torch.long, device=self.device,
        )
        self.mask = self.BUCKETS - 1
        self.ctx_counts = [
            torch.zeros(self.BUCKETS, dtype=torch.int32, device=self.device)
            for _ in range(6)
        ]
        self.full_counts = [
            torch.zeros(self.BUCKETS, dtype=torch.int32, device=self.device)
            for _ in range(6)
        ]

    @torch.no_grad()
    def update(self, tokens: Tensor) -> None:
        """Count n-gram occurrences from a 1D token stream."""
        t = tokens.to(device=self.device, dtype=torch.long).reshape(-1)
        n = t.numel()
        if n == 0:
            return
        self.total_tokens += n
        for oi, order in enumerate(range(self.min_order, self.max_order + 1)):
            if n < order:
                continue
            cw = order - 1
            length = n - order + 1
            ctx_hash = torch.zeros(length, dtype=torch.long, device=self.device)
            for k in range(cw):
                ctx_hash.bitwise_xor_(t[k : k + length] * self.primes[k])
            ctx_key = ctx_hash & self.mask
            full_key = (ctx_hash ^ (t[order - 1 : order - 1 + length] * self.primes[cw])) & self.mask
            ones = torch.ones(length, dtype=torch.int32, device=self.device)
            self.ctx_counts[oi].scatter_add_(0, ctx_key, ones)
            self.full_counts[oi].scatter_add_(0, full_key, ones)

    @torch.no_grad()
    def ngram_probs(self, x_batch: Tensor, y_batch: Tensor) -> tuple[Tensor, Tensor]:
        """Compute per-order n-gram probabilities for target tokens.

        Returns:
            order_p: (B, T, 6) — probability estimates per order
            order_valid: (B, T, 6) — whether each order has sufficient counts
        """
        bsz, slen = x_batch.shape
        dev = x_batch.device
        x = x_batch.long()
        y = y_batch.long()

        order_p = torch.full((bsz, slen, 6), 1.0 / self.V, device=dev)
        order_valid = torch.zeros(bsz, slen, 6, dtype=torch.bool, device=dev)

        for oi_rev in range(5, -1, -1):
            order = oi_rev + 2
            cw = order - 1
            if slen < cw:
                continue
            ctx_hash = torch.zeros(bsz, slen, dtype=torch.long, device=dev)
            for k in range(cw):
                shift = cw - 1 - k
                if shift > 0:
                    ctx_hash[:, shift:].bitwise_xor_(x[:, : slen - shift] * self.primes[k])
                else:
                    ctx_hash.bitwise_xor_(x * self.primes[k])
            ctx_key = (ctx_hash & self.mask).long()
            full_key = ((ctx_hash ^ (y * self.primes[cw])) & self.mask).long()
            ctx_c = self.ctx_counts[oi_rev][ctx_key.reshape(-1)].float().reshape(bsz, slen)
            full_c = self.full_counts[oi_rev][full_key.reshape(-1)].float().reshape(bsz, slen)
            p = torch.minimum(full_c, ctx_c) / ctx_c.clamp(min=1.0)
            p = p.clamp(0.0, 1.0)
            valid = ctx_c >= 2
            if cw > 0:
                valid[:, :cw] = False
            order_p[..., oi_rev] = torch.where(valid, p, order_p[..., oi_rev])
            order_valid[..., oi_rev] = valid

        return order_p, order_valid


def ngram_mixer_loss(
    gate_logits: Tensor,
    neural_logits: Tensor,
    target_ids: Tensor,
    order_p: Tensor,
    order_valid: Tensor,
    neural_floor: float = 0.05,
) -> Tensor:
    """Compute mixed NLL from neural + n-gram expert probabilities.

    Args:
        gate_logits: (B, T, 7) from the gate head — 1 neural + 6 n-gram orders
        neural_logits: (B*T, V) raw logits from the model
        target_ids: (B, T) target token ids
        order_p: (B, T, 6) n-gram probabilities per order
        order_valid: (B, T, 6) validity mask per order
    """
    bsz, slen = target_ids.shape

    # Neural probability for the correct token
    neural_lp = F.log_softmax(neural_logits.float(), dim=-1)
    neural_p = neural_lp.gather(1, target_ids.reshape(-1, 1)).squeeze(1).exp()
    neural_p = neural_p.reshape(bsz, slen)

    # Stack expert probabilities: [neural, order_2, ..., order_7]
    expert_p = torch.cat([neural_p.unsqueeze(-1), order_p], dim=-1)  # (B, T, 7)
    valid_mask = torch.cat([
        torch.ones(bsz, slen, 1, device=gate_logits.device, dtype=torch.bool),
        order_valid,
    ], dim=-1)  # (B, T, 7)

    # Masked softmax over gate logits
    masked_logits = gate_logits.masked_fill(~valid_mask, -1e9)
    weights = F.softmax(masked_logits, dim=-1)

    # Floor the neural weight at neural_floor to prevent collapse
    neural_w = neural_floor + (1.0 - neural_floor) * weights[..., :1]
    other_w = (1.0 - neural_floor) * weights[..., 1:]
    weights = torch.cat([neural_w, other_w], dim=-1)

    mixed_p = (weights * expert_p).sum(dim=-1)
    return -torch.log(mixed_p.clamp(min=1e-12)).mean()


# =============================================================================
# int6 quantization
# =============================================================================
def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    orig_shape = t32.shape
    if t32.ndim > 2:
        t32 = t32.view(t32.shape[0], -1)
    row_clip = t32.abs().amax(dim=1)
    scale = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -clip_range, clip_range).to(torch.int8)
    return q.view(orig_shape), scale


def dequantize_int6_per_row(q: Tensor, scale: Tensor) -> Tensor:
    orig_shape = q.shape
    q_flat = q.view(q.shape[0], -1).float()
    return (q_flat * scale.float()[:, None]).view(orig_shape)


def quantize_state_dict(state_dict: dict[str, Tensor], int6_keys: set[str] | None = None):
    """Quantize a state dict. Keys in int6_keys get int6, rest get fp16."""
    if int6_keys is None:
        int6_keys = {k for k in (state_dict.keys() if hasattr(state_dict, 'keys') else []) if "shared_moe.w_" in k and "router" not in k}
    quantized = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if name in int6_keys:
            q, s = quantize_int6_per_row(t)
            quantized[name + ".q"] = q
            quantized[name + ".s"] = s
        else:
            quantized[name] = t.to(torch.float16)
    return quantized


def dequantize_state_dict(quantized: dict[str, Tensor], int6_keys: set[str] | None = None):
    """Reverse of quantize_state_dict."""
    if int6_keys is None:
        # Infer int6 keys from the quantized dict (they have .q/.s suffixes)
        int6_keys = {k[:-2] for k in quantized.keys() if k.endswith(".q")}
    state_dict = {}
    for name in int6_keys:
        q = quantized[name + ".q"]
        s = quantized[name + ".s"]
        state_dict[name] = dequantize_int6_per_row(q, s)
    for name, tensor in quantized.items():
        if not name.endswith(".q") and not name.endswith(".s"):
            state_dict[name] = tensor.float()
    return state_dict


# =============================================================================
# Validation — competition-accurate val_bpb
# =============================================================================
def load_tokenizer_byte_tables(tokenizer_path: str, vocab_size: int):
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    base_bytes = torch.zeros(vocab_size, dtype=torch.int32)
    has_leading_space = torch.zeros(vocab_size, dtype=torch.bool)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        decoded = sp.Decode([i])
        base_bytes[i] = len(decoded.encode("utf-8"))
        has_leading_space[i] = piece.startswith("\u2581")
        is_boundary[i] = i in (sp.bos_id(), sp.eos_id(), sp.unk_id())
    return base_bytes, has_leading_space, is_boundary


@torch.no_grad()
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len: int | None = None,
    mixer: BackoffNgramMixer | None = None,
) -> tuple[float, float]:
    """Compute val_loss (nats) and val_bpb (bits/byte) on validation tokens.
    If mixer is provided and model has ngram_gate, uses mixed predictions."""
    if seq_len is None:
        seq_len = args.train_seq_len
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    use_mixer = mixer is not None and raw_model.ngram_gate is not None and mixer.total_tokens > 0

    bb = base_bytes_lut.to(device)
    hls = has_leading_space_lut.to(device)
    isb = is_boundary_token_lut.to(device)

    acc_dtype = torch.float32 if device.type == "mps" else torch.float64
    total_loss = torch.zeros((), device=device, dtype=acc_dtype)
    total_tokens = torch.zeros((), device=device, dtype=acc_dtype)
    total_bytes = torch.zeros((), device=device, dtype=acc_dtype)

    batch_tokens = args.val_batch_size
    local_batch_seqs = batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len

    for batch_start in range(0, total_seqs, local_batch_seqs):
        batch_end = min(batch_start + local_batch_seqs, total_seqs)
        raw_start = batch_start * seq_len
        raw_end = batch_end * seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        batch_token_count = y.numel()

        if use_mixer:
            logits, gate_logits = raw_model.forward_with_gate(x)
            # Compute mixed NLL using n-gram probabilities
            order_p, order_valid = mixer.ngram_probs(x, y)
            bsz, slen = y.shape
            neural_lp = F.log_softmax(logits.float(), dim=-1)
            neural_p = neural_lp.gather(2, y.unsqueeze(2)).squeeze(2)
            neural_p = neural_p.exp()
            expert_p = torch.cat([neural_p.unsqueeze(-1), order_p], dim=-1)
            valid_mask = torch.cat([
                torch.ones(bsz, slen, 1, device=device, dtype=torch.bool),
                order_valid,
            ], dim=-1)
            masked = gate_logits.masked_fill(~valid_mask, -1e9)
            weights = F.softmax(masked, dim=-1)
            neural_w = 0.05 + 0.95 * weights[..., :1]
            other_w = 0.95 * weights[..., 1:]
            weights = torch.cat([neural_w, other_w], dim=-1)
            mixed_p = (weights * expert_p).sum(dim=-1)
            batch_nll = -torch.log(mixed_p.clamp(min=1e-12))
            batch_loss = batch_nll.sum() / batch_token_count
        else:
            logits = raw_model(x)
            batch_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        total_loss += batch_loss.to(acc_dtype) * batch_token_count
        total_tokens += batch_token_count

        # Byte counting
        token_bytes = bb[y].to(dtype=torch.int64)
        prev_ids = torch.cat([x[:, -1:], y[:, :-1]], dim=1)
        space_adj = (hls[y] & ~isb[prev_ids]).to(dtype=torch.int64)
        total_bytes += (token_bytes + space_adj).to(acc_dtype).sum()

    val_loss = total_loss / total_tokens
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = total_tokens.item() / total_bytes.item()

    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# =============================================================================
# Learning rate schedule
# =============================================================================
def get_lr_multiplier(step: int, warmup: int, total: int, warmdown: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    if step >= total - warmdown:
        progress = (step - (total - warmdown)) / warmdown
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0


# =============================================================================
# Main training loop
# =============================================================================
def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    # --- Distributed setup ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_process = rank == 0

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    def log0(msg: str) -> None:
        if master_process:
            print(msg)

    log0(f"device={device} world_size={world_size} grad_accum_steps={grad_accum_steps}")

    # --- Logging ---
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"

    def log(msg: str) -> None:
        if not master_process:
            return
        print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    # --- Seed ---
    torch.manual_seed(args.seed + rank)

    # --- Data ---
    log0(f"Loading training data from {args.train_files}")
    train_stream = TokenStream(args.train_files)

    log0(f"Loading validation data from {args.val_files}")
    val_files = sorted(glob.glob(args.val_files))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files])
    log0(f"Val tokens: {val_tokens.numel():,}")

    log0(f"Loading tokenizer from {args.tokenizer_path}")
    base_bytes_lut, has_leading_space_lut, is_boundary_lut = load_tokenizer_byte_tables(
        args.tokenizer_path, args.vocab_size
    )

    # --- N-gram mixer (frozen oracle) ---
    mixer = None
    if args.ngram_enabled:
        log0("Prefilling n-gram oracle from training shards...")
        t_prefill = time.perf_counter()
        mixer = BackoffNgramMixer(
            vocab_size=args.vocab_size, device=str(device),
            buckets=args.ngram_buckets,
        )
        PREFILL_CHUNK = 10_000_000
        for shard_path in sorted(glob.glob(args.train_files)):
            raw = np.fromfile(shard_path, dtype="<u2")
            for off in range(0, len(raw), PREFILL_CHUNK):
                chunk = torch.from_numpy(
                    raw[off : off + PREFILL_CHUNK].astype(np.int32)
                ).to(device)
                mixer.update(chunk)
                del chunk
            del raw
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log0(f"  Prefilled {mixer.total_tokens:,} tokens in {time.perf_counter() - t_prefill:.1f}s")

    # --- Model ---
    model = GolfModel(
        vocab_size=args.vocab_size,
        dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        num_experts=args.num_experts,
        mlp_mult=args.mlp_mult,
        top_k=args.top_k,
        expert_depth=args.expert_depth,
        max_seq_len=args.train_seq_len,
        tie_embeddings=args.tie_embeddings,
        rope_base=args.rope_base,
        ngram_enabled=args.ngram_enabled,
        ngram_mixer_loss_weight=args.ngram_mixer_loss_weight,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log0(f"Model params: {total_params:,}")

    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if distributed else model

    # --- Optimizers: Muon for matrices, AdamW for scalars ---
    matrix_params = [
        raw_model.shared_moe.w_up,
        raw_model.shared_moe.w_down,
        raw_model.attn_qkv.weight,
        raw_model.attn_proj.weight,
    ]
    if hasattr(raw_model.shared_moe, "w_mid"):
        matrix_params.extend(list(raw_model.shared_moe.w_mid))
    scalar_params = [
        raw_model.embed.weight,
        raw_model.norm.weight,
        raw_model.shared_moe.router.weight,
    ]
    if not args.tie_embeddings and hasattr(raw_model, "lm_head"):
        scalar_params.append(raw_model.lm_head.weight)
    if raw_model.ngram_gate is not None:
        scalar_params.extend([raw_model.ngram_gate.weight, raw_model.ngram_gate.bias])

    opt_muon = Muon(matrix_params, lr=args.muon_lr, momentum=args.muon_momentum, backend_steps=5)
    opt_adam = torch.optim.AdamW(scalar_params, lr=args.adam_lr, betas=(0.9, 0.95))

    # --- Batch geometry ---
    local_batch_tokens = args.train_batch_tokens // world_size
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    micro_batch_seqs = local_batch_seqs // grad_accum_steps
    if micro_batch_seqs < 1:
        micro_batch_seqs = 1
        grad_accum_steps = local_batch_seqs

    log0(
        f"Batch: {args.train_batch_tokens} tokens total, "
        f"{local_batch_seqs} seqs/gpu, "
        f"{micro_batch_seqs} seqs/micro, "
        f"{grad_accum_steps} accum steps"
    )

    # --- Training loop ---
    log0(f"Training for up to {args.iterations} iterations (max {args.max_wallclock_seconds}s)...")
    t_start = time.perf_counter()
    total_tokens_seen = 0

    for step in range(1, args.iterations + 1):
        # Wallclock check
        elapsed = time.perf_counter() - t_start
        if elapsed >= args.max_wallclock_seconds:
            log(f"Wallclock limit reached at step {step} ({elapsed:.0f}s)")
            break

        # LR schedule
        lr_mult = get_lr_multiplier(step - 1, args.warmup_steps, args.iterations, args.warmdown_iters)
        for pg in opt_muon.param_groups:
            pg["lr"] = args.muon_lr * lr_mult
        for pg in opt_adam.param_groups:
            pg["lr"] = args.adam_lr * lr_mult

        # Forward/backward with gradient accumulation
        opt_muon.zero_grad()
        opt_adam.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            tokens = train_stream.take(micro_batch_seqs * args.train_seq_len + 1).to(device)
            x = tokens[:-1].reshape(micro_batch_seqs, args.train_seq_len)
            y = tokens[1:].reshape(micro_batch_seqs, args.train_seq_len)

            # Compute n-gram probabilities (frozen, no grad)
            ngram_kw = {}
            if mixer is not None and mixer.total_tokens > 0:
                with torch.no_grad():
                    order_p, order_valid = mixer.ngram_probs(x, y)
                ngram_kw = dict(ngram_order_p=order_p, ngram_order_valid=order_valid)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y, **ngram_kw) * grad_scale
            else:
                loss = model(x, y, **ngram_kw) * grad_scale
            loss.backward()
            accum_loss += loss.item()
            total_tokens_seen += y.numel() * world_size

        # Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        opt_muon.step()
        opt_adam.step()

        # Logging
        if step % args.train_log_every == 0:
            elapsed = time.perf_counter() - t_start
            tok_s = total_tokens_seen / elapsed
            log(f"step {step} | loss {accum_loss:.4f} | lr {lr_mult * args.muon_lr:.2e} | {tok_s:,.0f} tok/s | {elapsed:.0f}s")

        # Periodic validation
        if args.val_loss_every > 0 and step % args.val_loss_every == 0:
            val_loss, val_bpb = eval_val(
                args, raw_model, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_lut,
                mixer=mixer,
            )
            log(f"  val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")

    elapsed = time.perf_counter() - t_start
    log(f"Training complete in {elapsed:.1f}s, {total_tokens_seen:,} tokens seen")

    # --- Save raw model ---
    if master_process:
        torch.save(raw_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size: {code_bytes} bytes")

    # --- Quantize & compress ---
    quant_sd = quantize_state_dict(raw_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_sd, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)

    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        total_bytes = quant_file_bytes + code_bytes
        log(f"Serialized model int6+zlib: {quant_file_bytes} bytes")
        log(f"Total submission size: {total_bytes} bytes (cap: 16,000,000)")
        if total_bytes > 16_000_000:
            log(f"WARNING: exceeds 16MB cap by {total_bytes - 16_000_000} bytes!")

    # --- Roundtrip eval: load compressed artifact, rebuild model, eval ---
    log0("Roundtrip evaluation...")
    t_qeval = time.perf_counter()

    quant_loaded = torch.load(io.BytesIO(zlib.decompress(quant_blob)), weights_only=True)
    roundtrip_sd = dequantize_state_dict(quant_loaded)

    rt_model = GolfModel(
        vocab_size=args.vocab_size,
        dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        num_experts=args.num_experts,
        mlp_mult=args.mlp_mult,
        top_k=args.top_k,
        max_seq_len=args.train_seq_len,
        tie_embeddings=args.tie_embeddings,
        rope_base=args.rope_base,
        ngram_enabled=args.ngram_enabled,
    ).to(device)
    rt_model.load_state_dict(roundtrip_sd, strict=False)

    q_val_loss, q_val_bpb = eval_val(
        args, rt_model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_lut,
        mixer=mixer,
    )

    eval_time_ms = 1000.0 * (time.perf_counter() - t_qeval)
    log(f"final_int6_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{eval_time_ms:.0f}ms")
    log(f"final_int6_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
