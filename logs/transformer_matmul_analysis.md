# Transformer Forward Pass Matrix Multiplication Analysis

## Model Configuration
- Batch size: `B`
- Sequence length: `S` 
- Model dimension: `d_model`
- Number of heads: `H`
- Head dimension: `d_k = d_v = d_model / H`
- FFN inner dimension: `d_ff`
- Vocabulary size: `V`
- Number of layers: `L`

## Matrix Multiplications per Forward Pass

### 1. Token Embedding
- **Operation**: Token lookup (no matrix multiplication)
- **Input**: `(B, S)` token indices
- **Output**: `(B, S, d_model)` embeddings

### 2. Per Transformer Layer (repeated L times)

#### 2.1 Multi-Head Self-Attention
H * d_k = H * d_v = d_model
- **Q projection**: `(B, S, d_model) @ (H * d_k, d_model) = (B, S, d_model) @ (d_model, d_model) = (B, S, d_model)`
  - **Dimensions**: `(B*S, d_model) * (d_model, d_model)`
  - **FLOPs**: `2 * B * S * d_model * d_model`

- **K projection**: `(B, S, d_model) @ (H * d_k, d_model) = (B, S, d_model) @ (d_model, d_model) = (B, S, d_model)`
  - **Dimensions**: `(B*S, d_model) * (d_model, d_model)`
  - **FLOPs**: `2 * B * S * d_model * d_model`

- **V projection**: `(B, S, d_model) @ (H * d_v, d_model) = (B, S, d_model) @ (d_model, d_model) = (B, S, d_model)`
  - **Dimensions**: `(B*S, d_model) * (d_model, d_model)`
  - **FLOPs**: `2 * B * S * d_model * d_model`

- **Attention scores**: `Q @ K^T` where Q,K are `(B, H, S, d_k)`
  - **Dimensions**: `(B*H*S, d_k) * (d_k, S) = (B*H*S, S)`
  - **FLOPs**: `2 * B * H * S * d_k * S = 2 * B * H * S^2 * d_k`

- **Attention output**: `Attention_probs @ V` where V is `(B, H, S, d_v)`
  - **Dimensions**: `(B*H*S, S) * (S, d_v) = (B*H*S, d_v)`
  - **FLOPs**: `2 * B * H * S * S * d_v = 2 * B * H * S^2 * d_v`

- **Output projection**: `(B, S, d_model) @ (d_model, d_model) = (B, S, d_model)`
  - **Dimensions**: `(B*S, d_model) * (d_model, d_model)`
  - **FLOPs**: `2 * B * S * d_model * d_model`

#### 2.2 SwiGLU FFN
- **W1 projection**: `(B, S, d_model) @ (d_model, d_ff) = (B, S, d_ff)`
  - **Dimensions**: `(B*S, d_model) * (d_model, d_ff)`
  - **FLOPs**: `2 * B * S * d_model * d_ff`

- **W2 projection**: `(B, S, d_ff) @ (d_ff, d_model) = (B, S, d_model)`
  - **Dimensions**: `(B*S, d_ff) * (d_ff, d_model)`
  - **FLOPs**: `2 * B * S * d_ff * d_model`

- **W3 projection**: `(B, S, d_model) @ (d_model, d_ff) = (B, S, d_ff)`
  - **Dimensions**: `(B*S, d_model) * (d_model, d_ff)`
  - **FLOPs**: `2 * B * S * d_model * d_ff`


### 3. Final Language Modeling Head
- **LM head**: `(B, S, d_model) @ (d_model, V) = (B, S, V)`
  - **Dimensions**: `(B*S, d_model) * (d_model, V)`
  - **FLOPs**: `2 * B * S * d_model * V`

## Summary of Matrix Multiplications

### Per Layer (1 Transformer block):
1. **Q projection**: `(B*S, d_model) * (d_model, d_model)` → `2*B*S*d_model²`
2. **K projection**: `(B*S, d_model) * (d_model, d_model)` → `2*B*S*d_model²`
3. **V projection**: `(B*S, d_model) * (d_model, d_model)` → `2*B*S*d_model²`
4. **Attention scores**: `(B*H*S, d_k) * (d_k, S)` → `2*B*H*S²*d_k`
5. **Attention output**: `(B*H*S, S) * (S, d_v)` → `2*B*H*S²*d_v`
6. **Output projection**: `(B*S, d_model) * (d_model, d_model)` → `2*B*S*d_model²`
7. **FFN W1**: `(B*S, d_model) * (d_model, d_ff)` → `2*B*S*d_model*d_ff`
8. **FFN W3**: `(B*S, d_model) * (d_model, d_ff)` → `2*B*S*d_model*d_ff`
9. **FFN W2**: `(B*S, d_ff) * (d_ff, d_model)` → `2*B*S*d_ff*d_model`

### Total FLOPs per Layer:
```
Per layer = 8*B*S*d_model² + 2*B*H*S²*d_k + 2*B*H*S²*d_v + 6*B*S*d_model*d_ff
          = 8*B*S*d_model² + 2*B*H*S²*d_k + 2*B*H*S²*d_v + 6*B*S*d_model*d_ff
```

Since `d_k = d_v = d_model/H`, we have `H*d_k = H*d_v = d_model`, so:
```
Per layer = 8*B*S*d_model² + 4*B*S²*d_model + 6*B*S*d_model*d_ff
          = 2*B*S*d_model*(4*d_model + 2*S + 3*d_ff)
```

### Total FLOPs for entire model:
```
Total = L * 2*B*S*d_model*(4*d_model + 2*S + 3*d_ff) + 2*B*S*d_model*V
```

Where:
- L = number of layers
- The final term is the language modeling head

## Matrix Multiplication Count Summary
**Per Transformer Layer**: 9 matrix multiplications
**Total for L layers**: 9*L matrix multiplications  
**Plus LM head**: +1 matrix multiplication
**Grand Total**: **9*L + 1 matrix multiplications**







## GPT-2 Large Configuration
- vocab_size (V): 50,257
- context_length (S): 1,024
- num_layers (L): 48
- d_model: 1,600
- num_heads (H): 25
- d_ff: 6,400
- d_k = d_v = d_model/H = 1,600/25 = 64

## FLOP Calculations for GPT-2 Large (B=1, S=1024)

### Per Layer Calculations:

#### Attention Block:
1. **Q projection**: `2 * 1 * 1024 * 1600² = 5,242,880,000 FLOPs`
2. **K projection**: `2 * 1 * 1024 * 1600² = 5,242,880,000 FLOPs`
3. **V projection**: `2 * 1 * 1024 * 1600² = 5,242,880,000 FLOPs`
4. **Attention scores**: `2 * 1 * 25 * 1024² * 64 = 3,355,443,200 FLOPs`
5. **Attention output**: `2 * 1 * 25 * 1024² * 64 = 3,355,443,200 FLOPs`
6. **Output projection**: `2 * 1 * 1024 * 1600² = 5,242,880,000 FLOPs`

**Total Attention**: `4 × 5,242,880,000 + 2 × 3,355,443,200 = 27,682,406,400 FLOPs`

#### FFN Block:
7. **W1 projection**: `2 * 1 * 1024 * 1600 * 6400 = 20,971,520,000 FLOPs`
8. **W3 projection**: `2 * 1 * 1024 * 1600 * 6400 = 20,971,520,000 FLOPs`
9. **W2 projection**: `2 * 1 * 1024 * 6400 * 1600 = 20,971,520,000 FLOPs`

**Total FFN**: `3 × 20,971,520,000 = 62,914,560,000 FLOPs`

### Per Layer Total:
**Single Layer**: `27,682,406,400 + 62,914,560,000 = 90,596,966,400 FLOPs`

### All Layers:
**48 Layers**: `48 × 90,596,966,400 = 4,348,654,387,200 FLOPs`

### Language Modeling Head:
**LM Head**: `2 * 1 * 1024 * 1600 * 50257 = 164,279,070,720 FLOPs`

### Grand Total:
**Total FLOPs**: `4,348,654,387,200 + 164,279,070,720 = 4,512,933,457,920 FLOPs`

## Summary for GPT-2 Large (Single Forward Pass, B=1, S=1024):
- **Attention per layer**: 27.7 billion FLOPs
- **FFN per layer**: 62.9 billion FLOPs  
- **Total per layer**: 90.6 billion FLOPs
- **All 48 layers**: 4.35 trillion FLOPs
- **LM head**: 164.3 billion FLOPs
- **Grand total**: **4.51 trillion FLOPs**

## Breakdown by Component:
- **Attention**: 27.7B × 48 = 1.33 trillion FLOPs (29.4%)
- **FFN**: 62.9B × 48 = 3.02 trillion FLOPs (66.9%)
- **LM Head**: 164.3B FLOPs (3.6%)
- **Total**: 4.51 trillion FLOPs (100%)


| quantity                                                     | symbol                                          | value                                      |
| ------------------------------------------------------------ | ----------------------------------------------- | ------------------------------------------ |
| forward-pass FLOPs/step for **B = 1, S = 1024** (your tally) | $F_{1}$                                         | **4.51 × 10¹²**                            |
| scale-up to **B = 1024** sequences                           | $F_{\text{fwd}}\;=\;1024 F_{1}$                 | 4.51 × 10¹² × 1024 ≈ **4.62 × 10¹⁵**       |
| add backward pass (× 2)                                      | $F_{\text{step}}\;=\;3 F_{\text{fwd}}$          | 3 × 4.62 × 10¹⁵ ≈ **1.39 × 10¹⁶**          |
| steps                                                        | $N_{\text{step}}$                               | 400 000                                    |
| **total training FLOPs**                                     | $F_{\text{tot}}=F_{\text{step}}N_{\text{step}}$ | 1.39 × 10¹⁶ × 4 × 10⁵ ≈ **5.5 × 10²¹**     |
| usable throughput on one A100 (50 % × 19.5 TF/s)             | $R$                                             | 9.75 × 10¹² FLOP s⁻¹                       |
| **time**                                                     | $t = F_{\text{tot}}/R$                          | 5.5 × 10²¹ / 9.75 × 10¹² ≈ **5.7 × 10⁸ s** |


Convert seconds → years

5.7 × 10⁸ s÷(3600×24×365) ≈ 18years.