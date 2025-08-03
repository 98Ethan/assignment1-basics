"""Attention mechanisms for transformer models."""

import math
import torch
import torch.nn as nn
from jaxtyping import Float, Bool
from einops import einsum, rearrange

from .layers import Linear, RotaryPositionalEmbedding


def softmax(tensor: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    """Apply the softmax operation on a tensor along the specified dimension.
    
    Args:
        tensor: torch.Tensor, input tensor of arbitrary shape
        dim: int, dimension along which to apply softmax
        
    Returns:
        torch.Tensor, output tensor with softmax applied along the specified dimension
    """
    # Subtract the maximum value for numerical stability
    max_vals = torch.max(tensor, dim=dim, keepdim=True)[0]
    shifted = tensor - max_vals
    
    # Compute exponentials
    exp_vals = torch.exp(shifted) # shape: same as input tensor
    
    # Compute sum along the specified dimension
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)
    
    # Return normalized probabilities
    return exp_vals / sum_exp


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... queries d_k"],
    K: Float[torch.Tensor, "... keys d_k"], 
    V: Float[torch.Tensor, "... keys d_v"],
    mask: Bool[torch.Tensor, "queries keys"] | None = None,
) -> Float[torch.Tensor, "... queries d_v"]:
    """Implement scaled dot-product attention.
    
    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., values, d_v)
        mask: Optional boolean mask of shape (..., queries, keys).
              Positions with mask value True should collectively sum to 1,
              positions with mask value False should be zero.
              
    Returns:
        Output tensor of shape (..., queries, d_v)
    """
    
    # Get the key dimension for scaling
    d_k = Q.shape[-1]
    scaling_factor = math.sqrt(d_k)
    
    # Compute scaled attention scores: QK^T / sqrt(d_k)
    attention_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / scaling_factor
    
    # Apply mask by setting masked positions to negative infinity
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

    # Compute attention probabilities via softmax
    attention_probs = softmax(attention_scores, dim=-1)

    # Apply attention to values and return weighted sum
    context = einsum(attention_probs, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return context


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta=None, max_seq_len=None, device=None, dtype=None):
        """Construct a causal multi-head self-attention module.
        
        Args:
            d_model: int, Dimensionality of the Transformer block inputs
            num_heads: int, Number of heads to use in multi-head self-attention
            theta: float | None, Θ value for RoPE, if None, RoPE is not applied
            device: torch.device | None = None, Device to store the parameters on
            dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Create linear projections for Q, K, V, and output
        self.num_heads = num_heads
        d_k = d_v = d_model // num_heads  # Dimension of each head

        self.q_proj = Linear(in_features=d_model, out_features=num_heads * d_k, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=num_heads * d_k, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=num_heads * d_v, device=device, dtype=dtype)
        self.o_proj = Linear(in_features=num_heads * d_v, out_features=d_model, device=device, dtype=dtype)

        # If theta is provided, use RoPE
        self.rope = None
        if theta and max_seq_len:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Apply causal multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """

        # Project inputs to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Rearrange Q, K, V to (batch, num_heads, seq_len, d_k)
        Q = rearrange(Q, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        K = rearrange(K, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        V = rearrange(V, "b s (h d_v) -> b h s d_v", h=self.num_heads)

        # Apply RoPE: RoPE is typically applied to Q and K, not V
        seq_len = Q.shape[-2]

        if self.rope:
            token_positions = torch.arange(seq_len, device=Q.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
            
        # Compute attention mask for causal self-attention
        # In self-attention, queries = keys = seq_len
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device))

        # Scaled dot-product attention: Use rotated Q, K with original V
        context = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Rearrange context back to (batch, seq_len, d_model)
        context = rearrange(context, "b h s d_v -> b s (h d_v)")

        # Project back to d_model
        output = self.o_proj(context)

        return output