"""Utility functions for transformer training."""

import math
import os
import typing
import torch
import numpy as np
from collections.abc import Iterable


def get_lr_cosine_schedule(
    it: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """Get learning rate at iteration t according to cosine schedule with warmup.
    
    Args:
        it: Current iteration number
        alpha_max: Maximum learning rate (after warmup)
        alpha_min: Minimum learning rate (final value)
        T_w: Number of warmup iterations  
        T_c: Number of cosine annealing iterations (as T_c > T_w)
        
    Returns:
        Learning rate at the given iteration
    """
    if it < T_w:
        # Linear warmup phase
        return alpha_max * it / T_w
    elif it <= T_c:
        # Cosine annealing phase  
        progress = (it - T_w) / (T_c - T_w)
        return alpha_min + (alpha_max - alpha_min) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        # After cosine cycle, stay at minimum
        return alpha_min


@torch.no_grad()  # ← autograd is off inside this function
def gradient_clipping(
    params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6
) -> None:
    """Clip gradients so that the global L2 norm ≤ max_l2_norm (in-place)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    # Compute global L2 norm in fp32 for numerical stability.
    device = grads[0].device
    norms = torch.stack(
        [g.float().norm() for g in grads]
    )  # 1-D tensor of per-tensor norms
    total_norm = torch.linalg.vector_norm(norms, 2).to(device)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


def get_batch(
    dataset: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample language modeling input sequences and their corresponding labels from the dataset.
    
    Args:
        dataset: 1D numpy array of integer token IDs in the dataset
        batch_size: Desired batch size to sample
        context_length: Desired context length of each sampled example
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
                to place the sampled input sequences and labels on
    
    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    assert dataset.ndim == 1 and dataset.dtype.kind in "iu", "dataset must be 1-D integer array"

    n = dataset.shape[0]
    assert n >= context_length, "n must be at least context_length"

    rng = np.random.default_rng()
    starts = rng.integers(0, n - context_length, size=batch_size)  # uniform sampling

    # Build input and label sequences
    X = np.stack([dataset[i : i + context_length]         for i in starts])
    Y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])

    # To torch on requested device
    X = torch.as_tensor(X, dtype=torch.long, device=device)
    Y = torch.as_tensor(Y, dtype=torch.long, device=device)
    return X, Y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """Save checkpoint containing model, optimizer states and iteration number."""
    
    obj = {
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load checkpoint and restore model and optimizer states. Returns iteration number."""
    obj = torch.load(src, map_location='cpu')
    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optimizer_state"])
    return obj["iteration"]


if __name__ == "__main__":

    """ test for get_batch function."""
    from pathlib import Path
    from cs336_basics.tokenize.io_utils import load_token_file_mmap

    token_file_path = Path("./data") / "tokens" / "TinyStoriesV2-GPT4-train.bin"

    # Convert to numpy array
    dataset = load_token_file_mmap(token_file_path)

    batch_size = 5
    context_length = 10
    device = "cpu"

    X, Y = get_batch(dataset, batch_size, context_length, device)
