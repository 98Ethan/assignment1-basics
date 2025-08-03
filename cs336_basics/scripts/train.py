#!/usr/bin/env python3
"""Training script for Transformer language model."""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import numpy as np
from einops import rearrange


from cs336_basics.transformer.model import TransformerLM
from cs336_basics.transformer.optimizers import AdamW
from cs336_basics.transformer.loss import cross_entropy_loss
from cs336_basics.transformer.utils import (
    get_lr_cosine_schedule, 
    gradient_clipping, 
    get_batch,
    save_checkpoint,
    load_checkpoint
)
from cs336_basics.scripts.experiment_logging import ExperimentLogger
from cs336_basics.tokenize.io_utils import load_token_file_mmap


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Transformer language model")
    
    # Data paths
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data binary file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data binary file")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=512, help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--theta", type=float, default=10000.0, help="RoPE theta parameter")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_iters", type=int, default=10000, help="Maximum number of training iterations")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--eval_iters", type=int, default=200, help="Number of iterations for evaluation")
    
    # Optimizer hyperparameters
    parser.add_argument("--lr_max", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--lr_min", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000, help="Number of cosine annealing iterations")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-training", help="W&B project name")
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Get the appropriate device."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def estimate_loss(
    model: TransformerLM,
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """Estimate loss on training and validation data."""
    model.eval()
    losses = {}
    
    for split_name, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(data, batch_size, context_length, device)
            with torch.no_grad():
                logits = model(X)
                loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), Y.view(-1))
                split_losses.append(loss.item())
        losses[split_name] = np.mean(split_losses)
    
    model.train()
    return losses


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Setup experiment logging
    experiment_logger = ExperimentLogger(
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        config=vars(args),
        log_level=args.log_level
    )
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load datasets with memory mapping
    logger.info(f"Loading training data from {args.train_data}")
    train_data = load_token_file_mmap(args.train_data)
    logger.info(f"Training data shape: {train_data.shape}")
    
    logger.info(f"Loading validation data from {args.val_data}")
    val_data = load_token_file_mmap(args.val_data)
    logger.info(f"Validation data shape: {val_data.shape}")
    
    # Initialize model
    logger.info("Initializing model")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=device
    )
    
    # Count parameters and log model info
    n_params = sum(p.numel() for p in model.parameters())
    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "theta": args.theta
    }
    experiment_logger.log_model_info(n_params, model_config)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # Setup checkpointing
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_iter = 0
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        logger.info(f"Resumed from iteration {start_iter}")
    
    # Training loop
    logger.info("Starting training")
    model.train()
    
    # Start experiment timing
    experiment_logger.start_training()
    
    for iter_num in range(start_iter, args.max_iters):
        # Get learning rate
        lr = get_lr_cosine_schedule(
            iter_num, 
            args.lr_max, 
            args.lr_min, 
            args.warmup_iters, 
            args.cosine_cycle_iters
        )
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        # Forward pass
        logits = model(X) # (batch_size, seq_len, vocab_size)

        logits_flat  = rearrange(logits, 'b s v -> (b s) v')  # (batch_size * seq_len, vocab_size)
        targets_flat = rearrange(Y,      'b s   -> (b s)')
        loss = cross_entropy_loss(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iter_num % args.log_interval == 0:
            experiment_logger.log_training_step(
                iter_num=iter_num,
                loss=loss.item(),
                lr=lr,
                log_interval=args.log_interval,
                batch_size=args.batch_size,
                context_length=args.context_length
            )
        
        # Evaluation
        if iter_num % args.eval_interval == 0:
            eval_start_time = time.time()
            logger.info("Running evaluation...")
            losses = estimate_loss(
                model, train_data, val_data, 
                args.batch_size, args.context_length, 
                args.eval_iters, device
            )
            eval_time = time.time() - eval_start_time
            
            experiment_logger.log_evaluation(
                iter_num=iter_num,
                train_loss=losses['train'],
                val_loss=losses['val'],
                eval_time=eval_time
            )
        
        # Save checkpoint
        if iter_num % args.save_interval == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_{iter_num}.pt"
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            experiment_logger.log_checkpoint_save(iter_num, str(checkpoint_path))
    
    # Final checkpoint
    final_checkpoint_path = Path(args.checkpoint_dir) / "final_checkpoint.pt"
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    experiment_logger.log_checkpoint_save(args.max_iters, str(final_checkpoint_path))
    
    # Final evaluation
    logger.info("Running final evaluation...")
    losses = estimate_loss(
        model, train_data, val_data,
        args.batch_size, args.context_length,
        args.eval_iters, device
    )
    
    experiment_logger.log_final_results(
        train_loss=losses['train'],
        val_loss=losses['val']
    )
    
    # Clean up logging
    experiment_logger.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()