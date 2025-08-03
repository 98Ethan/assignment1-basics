"""Loss functions for transformer models."""

import torch
from jaxtyping import Float, Int


def cross_entropy_loss(
    logits: Float[torch.Tensor, "batch_size vocab_size"],
    targets: Int[torch.Tensor, "batch_size"],
) -> Float[torch.Tensor, ""]:
    """Compute cross entropy loss with numerical stability.
    
    Args:
        logits: Predicted logits of shape (batch_size, vocab_size)
        targets: Target class IDs of shape (batch_size,)
        
    Returns:
        Scalar tensor with average cross entropy loss across the batch
    """
    batch_size = logits.shape[0]
    
    # Compute log-softmax using logsumexp for numerical stability
    log_softmax = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    
    # Extract log probabilities for target tokens
    batch_indices = torch.arange(batch_size, device=logits.device)
    target_log_probs = log_softmax[batch_indices, targets]
    
    # Return negative mean (cross entropy loss)
    loss = -torch.mean(target_log_probs)
    
    return loss