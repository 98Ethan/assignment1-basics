"""Transformer model architecture."""

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from .layers import Embedding, RMSNorm, Linear, SwiGLU
from .attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta=None, max_seq_len=None, device=None, dtype=None):
        """Construct a pre-norm Transformer block.
        
        Args:
            d_model: int, Dimensionality of the Transformer block inputs
            num_heads: int, Number of heads to use in multi-head self-attention
            d_ff: int, Dimensionality of the position-wise feed-forward inner layer
            theta: float | None, Θ value for RoPE, if None, RoPE is not applied
            max_seq_len: int | None, Maximum sequence length for RoPE
            device: torch.device | None = None, Device to store the parameters on
            dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()
        
        # Layer normalization modules
        self.attn_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            theta=theta, 
            max_seq_len=max_seq_len, 
            device=device, 
            dtype=dtype
        )
        
        # Feed-forward network
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
    
    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Apply pre-norm Transformer block transformation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm attention with residual connection
        attn_input = self.attn_norm(x)
        attn_output = self.attn(attn_input)
        x = x + attn_output
        
        # Pre-norm FFN with residual connection
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output
        
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None,
        dtype=None
    ):
        """Construct a Transformer language model.
        
        Args:
            vocab_size: int, Size of the vocabulary
            context_length: int, Maximum context length (for position embeddings)
            d_model: int, Dimensionality of the model
            num_layers: int, Number of Transformer blocks to use
            num_heads: int, Number of heads to use in multi-head self-attention
            d_ff: int, Dimensionality of the position-wise feed-forward inner layer
            theta: float, Θ value for RoPE (default: 10000.0)
            device: torch.device | None = None, Device to store the parameters on
            dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()
        
        # Token embedding
        self.token_embedding = Embedding( # vocab_id to embedding(d_model)
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        # Language modeling head (output projection)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )
    
    def forward(self, token_ids: Int[torch.Tensor, "batch_size sequence_length"]) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        """Apply the Transformer language model to input token IDs.
        
        Args:
            token_ids: Input token IDs of shape (batch_size, sequence_length)
            
        Returns:
            Logits over vocabulary of shape (batch_size, sequence_length, vocab_size)
        """

        # Token embeddings
        x: Float[torch.Tensor, "batch_size seq_len d_model"] = self.token_embedding(token_ids)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x) # (batch_size, seq_len, d_model)
        
        # Language modeling head
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)
        
        return logits