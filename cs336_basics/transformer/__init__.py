"""Transformer model implementation with modular components."""

# Import all classes and functions to maintain backward compatibility
from .layers import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding
from .attention import softmax, scaled_dot_product_attention, MultiHeadSelfAttention
from .model import TransformerBlock, TransformerLM
from .loss import cross_entropy_loss
from .optimizers import AdamW
from .utils import get_lr_cosine_schedule, gradient_clipping

__all__ = [
    # Layers
    "Linear",
    "Embedding", 
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    
    # Attention
    "softmax",
    "scaled_dot_product_attention",
    "MultiHeadSelfAttention",
    
    # Model
    "TransformerBlock",
    "TransformerLM",
    
    # Loss
    "cross_entropy_loss",
    
    # Optimizers
    "AdamW",
    
    # Utils
    "get_lr_cosine_schedule",
    "gradient_clipping",
]