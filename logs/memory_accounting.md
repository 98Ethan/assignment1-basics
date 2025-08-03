
# Problem (adamwAccounting): Resource accounting for training with AdamW *(SwiGLU edition)*

> **Assumption:** every tensor is stored in **float32** (4 bytes / element).  
> **Change:** the FFN uses **SwiGLU** – it has **three** weight matrices (`W1`, `W2`, `W3`).

---

time  ───────────────────────────────► \
        forward ..........│ backward .......... \
layer 1 → layer 2 → ... → layer L │  grad_L → ... → grad_2 → grad_1

Autograd needs enough information to compute
∂loss/∂x for every intermediate x.

During the forward pass it therefore stores the tensors (or a compressed
representation) that each backward op will later read

## Snapshot — what occupies GPU memory at peak

| Bucket | Elements (symbolic) | Purpose |
|--------|--------------------|---------|
| **Parameters** | `P` | Trainable weights |
| **Gradients** | `P` | One ∇ tensor per weight |
| **Optimizer state** | `2 P` | First & second moments (*m*, *v*) for AdamW |
| **Forward activations** | `A` | Saved tensors from the forward pass |
| **Total peak** | **`4 P + A`** | Right after backward starts |

*(multiply by 4 bytes → bytes; divide by 2³⁰ → GiB)*

---
*(Symbols: `b =batch_size`, `s =context_length`, `L =num_layers`, `d =d_model`, `V =vocab_size`)*

## 1 · Parameter count `P`

| Component | Elements per **layer** |
|-----------|------------------------|
| Multi‑head attention `Wq, Wk, Wv, Wo` | **`4 d²`** |
| **SwiGLU FFN** <br/>`W1 (d×4d)`, `W2 (4d×d)`, `W3 (d×4d)` | **`12 d²`** |
| RMSNorm (×2) | **`2 d`**  |

```
Per‑layer total          : 16 d² + 2 d
=> Stack of L layers     : L (16 d² + 2 d)
Embeddings               : V d
Final RMSNorm            : d

────────────────────────────────────
P = L (16 d² + 2 d) + V d + d
```

---

## 2 · Activation count `A`

What we must cache **per layer & per token**:

| Tensor kept | Elements |
|-------------|----------|
| Block input after RMSNorm | `b s d` |
| Q, K, V projections | `3 b s d` |
| Attention output | `b s d` |
| FFN `W1` pre‑SwiGLU | `4 b s d` |
| FFN `W3` linear output | `4 b s d` |
| FFN `W2` output (before residual) | `b s d` |

**Per layer total → `14 b s d`.**

Add the final RMSNorm output and the loss head (logits + soft‑max):

```
A = 14 L b s d  +  b s d  +  b s V
```

---

## 3 · Putting the pieces together

```
peak_elements = 4 P + A
              = 4 (L (16 d² + 2 d) + V d + d)
                + (14 L b s d + b s d + b s V)

peak_bytes    = 4 × peak_elements
peak_GiB      = peak_bytes / 2³⁰
```

---

### Memory‑pressure insights

* **`4 P`** (params + grads + Adam moments) is the long‑lived block.  
* **`A`** grows with batch × sequence; even larger now because SwiGLU doubles the big hidden tensor.

| Technique | Shrinks |
|-----------|---------|
| Mixed precision / FP8 | every bucket |
| Activation checkpointing | `A` |
| 8‑bit optimizer / offload | part of `2 P` |
| ZeRO parameter sharding | the entire `4 P` |






# GPT‑2 Large – Peak GPU Memory Estimate (FP32)

Configuration  
: *vocab* **V = 50,257** | *layers* **L = 48** | *d_model* **d = 1600** | *context* **s = 1024**

---

## 1. Parameter count `P`

| component | elements |
|-----------|----------|
| per‑layer (`16d² + 2d`) | 40,963,200 |
| × 48 layers | 1,966,233,600 |
| embeddings `Vd` | 80,411,200 |
| final RMSNorm `+d` | 1600 |
| **Total P** | **2,046,646,400** |

## 2. Long‑lived bucket `4P`

```
4P = 8,186,585,600 elements
```

## 3. Activation term `A`

Per batch element:

```
A_unit = s*d*(14L+1) + s*V
       = 1,154,106,368 elements
```

## 4. Peak memory for common batch sizes

| batch `b` | peak elements | peak bytes | **peak memory** |
|-----------|---------------|------------|-----------------|
| 1 | 9,340,691,968 | 37,362,767,872 | **34.80 GiB** |
| 2 | 10,494,798,336 | 41,979,193,344 | **39.10 GiB** |
| 4 | 12,803,011,072 | 51,212,044,288 | **47.69 GiB** |
| 8 | 17,419,436,544 | 69,677,746,176 | **64.89 GiB** |


*(Each float32 element = 4 bytes. 1 GiB = 2³⁰ bytes.)*

---
### Notes
* Reducing precision (FP16/BF16) halves all numbers.
* Activation checkpointing lowers the `A` term.
* Tying the output embedding would subtract `Vd` params + grads + 2Vd optimizer‑state.
