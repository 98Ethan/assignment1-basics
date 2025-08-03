"""Basic neural network layers for transformer models."""

import math
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """Construct a linear transformation module.
        
        Args:
            in_features: int, final dimension of the input
            out_features: int, final dimension of the output
            device: torch.device | None = None, Device to store the parameters on
            dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()
        
        # Create weight parameter as W (not W^T) for memory ordering reasons
        self.weight: Float[torch.Tensor, "out_features in_features"] = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal
        # σ² = 2/(d_in + d_out), so σ = sqrt(2/(d_in + d_out))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input.
        
        Args:
            x: torch.Tensor, input tensor
            
        Returns:
            torch.Tensor, output after linear transformation
        """
        # return torch.matmul(x, self.weight.T)
        return einsum(x, self.weight, "... in_feat, out_feat in_feat -> ... out_feat")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """Construct an embedding module.
        
        Args:
            num_embeddings: int, Size of the vocabulary
            embedding_dim: int, Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None, Device to store the parameters on
            dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()

        # torch.LongTensor (vocab_size, d_model)
        self.embeddings: Float[torch.Tensor, "num_embeddings embedding_dim"] = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # Initialize embeddings using truncated normal
        nn.init.trunc_normal_(self.embeddings, mean=0.0, std=1, a=-3, b=3)
    
    def forward(self, token_ids: Int[torch.Tensor, "batch_size sequence_length"]) -> torch.Tensor:

        """Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids: torch.Tensor, input token IDs, shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor, embedding vectors for the token IDs
        """
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the RMSNorm module.
        
        Args:
            d_model: int, Hidden dimension of the model
            eps: float = 1e-5, Epsilon value for numerical stability
            device: torch.device | None = None, Device to store the parameters on
            dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()
        
        # Create a learnable parameter g (scale) for normalization
        self.gain = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        self.eps = eps
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor and return a tensor of the same shape.
        
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            torch.Tensor, normalized tensor of the same shape
        """
        # upcast input to torch.float32 to prevent overflow when you square
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Calculate the root mean square of the input tensor
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        result = x / rms * self.gain

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        """Construct a SwiGLU FFN module.
        
        Args:
            d_model: int, Hidden dimension of the model  
            d_ff: int | None, Hidden dimension of the FFN (typically 8/3 * d_model)
            device: torch.device | None = None, Device to store the parameters on
            dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()

        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
        
        # Create the three linear transformations using our Linear class
        self.w1: Linear = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)  # d_ff x d_model
        self.w2: Linear = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)    # d_model x d_ff
        self.w3: Linear = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)  # d_ff x d_model

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN transformation.
        
        Formula: FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1*x) ⊙ W3*x)
        where ⊙ denotes element-wise multiplication
        
        Args:
            x: torch.Tensor, input tensor of shape (..., d_model)
            
        Returns:
            torch.Tensor, output tensor of shape (..., d_model)
        """
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        result = self.w2(silu * self.w3(x))
        return result


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Construct the RoPE module and create buffers if needed.
        
        Args:
            theta: float, Θ value for the RoPE
            d_k: int, dimension of query and key vectors
            max_seq_len: int, Maximum sequence length that will be inputted
            device: torch.device | None = None, Device to store the buffer on
        """
        super().__init__()

        pos = torch.arange(max_seq_len, device=device)       # (max_seq_len,)
        dim_idx = torch.arange(0, d_k, 2, device=device)     # (d_k//2,)
    
        angles = pos[:, None] / (theta ** (dim_idx / d_k))   # (max_seq_len, d_k//2)
        sin = angles.sin()                                   # (max_seq_len, d_k//2) 
        cos = angles.cos()                                   # (max_seq_len, d_k//2)
        
        # Register as buffers
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor and return a tensor of the same shape.
        
        Note that you should tolerate x with an arbitrary number of batch dimensions.
        You should assume that the token positions are a tensor of shape (..., seq_len)
        specifying the token positions of x along the sequence dimension.
        
        You should use the token positions to slice your (possibly precomputed) cos and sin
        tensors along the sequence dimension.
        
        Args:
            x: torch.Tensor, input tensor of shape (..., seq_len, d_k)
            token_positions: torch.Tensor, token positions of shape (..., seq_len)
            
        Returns:
            torch.Tensor, tensor with RoPE applied, same shape as input
        """
        # Get cos/sin for the specific positions
        cos_vals = self.cos[token_positions]  # (..., seq_len, d_k//2)
        sin_vals = self.sin[token_positions]  # (..., seq_len, d_k//2)
        
        # Split x into even and odd dimensions
        x_even = x[..., ::2]   # (..., seq_len, d_k//2)
        x_odd = x[..., 1::2]   # (..., seq_len, d_k//2)
        
        # Apply RoPE rotation
        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals
        
        # Interleave back together using rearrange
        from einops import rearrange
        stacked = torch.stack([rotated_even, rotated_odd], dim=-1)
        return rearrange(stacked, "... d_k_half pair -> ... (d_k_half pair)")