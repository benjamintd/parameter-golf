# LeReMix: Depth-Recurrent Mixture of Experts

**val_bpb: TBD** | **~TBD MB** | 8xH100 SXM

> **Le**arned **Re**currence + **Mix**ture of Experts

## Approach

A Universal Transformer-inspired architecture where a single attention + MoE block
is shared across all depth steps, giving 12 virtual layers from 1 physical block.
The key innovations are:

1. **Passthrough Expert (Expert 0)** — a parameter-free identity expert that acts
   as implicit Adaptive Computation Time. The router learns to send "done" tokens
   to expert 0, skipping all matmuls and saving compute.

2. **Timestep Embeddings** — sinusoidal depth signal added at each recurrence step
   so the model (and router) can distinguish "which iteration am I in" and learn
   step-dependent behavior (e.g., coarse processing early, refinement late).

3. **SigReg Regularization** — random-projection sketch of the MoE input that
   penalizes mean drift and variance deviation from 1.0, preventing mode collapse
   in expert routing. Acts as a distributional prior on the latent space.

4. **Dual Optimizer** — Muon (Newton-Schulz orthogonalization) for weight matrices,
   AdamW for embeddings/router/norms. Muon keeps weight matrices well-conditioned
   across the depth recurrence.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 1 physical, 12 recurrent (shared weights) |
| Model dim | 512 |
| Attention | 8 heads, 64 head_dim |
| Positional encoding | RoPE (full) + sinusoidal timestep |
| MoE experts | 64 (1 passthrough + 63 learned, top-2) |
| MoE hidden dim | 1024 (2x expansion) |
| MoE activation | ReLU |
| Embeddings | Tied input/output |
| Normalization | RMSNorm (pre-norm) |
| Regularization | SigReg (random-projection distributional prior, anti-collapse) |
| Memory | Gradient checkpointing across recurrence |

### Parameter breakdown

| Component | Parameters | % |
|-----------|-----------|---|
| MoE w1 (63 experts) | 32,256 x 1024 = 32.3M | 47.8% |
| MoE w2 (63 experts) | 63 x 1024 x 512 = 32.3M | 47.8% |
| Attention QKV | 512 x 1536 = 786K | 1.2% |
| Attention proj | 512 x 512 = 262K | 0.4% |
| Embeddings | 1024 x 512 = 524K | 0.8% |
| Router | 512 x 64 = 33K | <0.1% |
| RMSNorm | 512 | <0.1% |
| **Total** | **~67.7M** | |

> Note: Expert 0 (passthrough) has zero parameters. Depth recurrence means
> these parameters are reused 12 times, giving an effective depth of 12 layers.

### How passthrough expert works

```
Router output: softmax over [expert_0, expert_1, ..., expert_63]
                                 |            |
                              identity    x @ w1 -> ReLU -> @ w2
                            (skip matmul)  (standard MLP)
```

Tokens routed to expert 0 pass through unchanged — zero compute, zero parameters.
The router learns when a token has "converged" at a given depth step and doesn't
need further transformation. This is analogous to ACT (Adaptive Computation Time)
but learned implicitly through the routing mechanism rather than an explicit
halting probability.

### How timestep embedding works

At each recurrence step `t`, a sinusoidal embedding is added to the hidden state:

```python
x = x + timestep_embed[t]  # (dim,) broadcast to (B, T, dim)
```

This allows the shared attention and MoE weights to behave differently at different
depths. Without this signal, every recurrence step sees identical inputs and cannot
learn step-specific behavior like "do coarse parsing at step 1, refine at step 12."

## Quantization

- MoE weight banks (w1, w2): **int6** per-row quantization (6 bits, [-31, 31])
- All other weights: **fp16**
- Compression: **zlib** level 9
- Roundtrip eval: decompress → dequantize → rebuild model → re-evaluate

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer (matrices) | Muon (lr=0.025, momentum=0.95, NS5 steps=5) |
| Optimizer (scalars) | AdamW (lr=0.003, betas=(0.9, 0.95)) |
| Batch tokens | 524,288 |
| Sequence length | 1024 |
| Gradient accumulation | 8 // world_size |
| Gradient clipping | 0.3 |
| LR schedule | Linear warmup (20 steps) + cosine warmdown |
| Wallclock limit | 600s |

## Run Command

```bash
RUN_ID=leremix \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Differences from standard Parameter Golf submissions

| Standard approach | LeReMix |
|-------------------|---------|
| N independent layers | 1 shared block, N recurrence steps |
| Fixed-depth FFN/MLP per layer | MoE with per-token expert routing |
| No halting mechanism | Passthrough expert = implicit ACT |
| Separate params per layer | Weight sharing + timestep embedding |
| ~15M params for 9-11 layers | ~67M params reused 12x |

## Novel contributions

- **Passthrough expert as implicit ACT**: To our knowledge, this is the first
  parameter golf submission to use a zero-parameter identity expert for learned
  adaptive halting in a depth-recurrent architecture.
- **Timestep-conditioned MoE routing**: The router sees different representations
  at different depths, enabling step-dependent expert selection without additional
  parameters.
- **Depth recurrence + MoE**: Combining Universal Transformer-style weight sharing
  with Mixture of Experts, where the expert bank is the primary parameter store
  and depth comes for free via recurrence.
