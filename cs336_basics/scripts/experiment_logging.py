"""Experiment logging infrastructure for tracking training runs."""

import time
from typing import Dict, Any, Optional
import logging


class ExperimentLogger:
    """Handles experiment tracking and logging for training runs."""
    
    def __init__(
        self,
        use_wandb: bool = False,
        wandb_project: str = "transformer-training",
        config: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO"
    ):
        """Initialize experiment logger.
        
        Args:
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            config: Configuration dictionary to log
            log_level: Python logging level
        """
        self.use_wandb = use_wandb
        self.wandb_logger = None
        self.training_start_time = None
        self.step_start_time = None
        
        # Setup standard logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize W&B if requested
        if self.use_wandb:
            self._init_wandb(wandb_project, config)
    
    def _init_wandb(self, project: str, config: Optional[Dict[str, Any]]):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            self.wandb_logger = wandb.init(
                project=project,
                config=config or {}
            )
            self.logger.info(f"Initialized W&B logging for project: {project}")
        except ImportError:
            self.logger.warning("W&B not available, skipping wandb logging")
            self.use_wandb = False
    
    def start_training(self):
        """Mark the start of training for timing."""
        self.training_start_time = time.time()
        self.step_start_time = time.time()
        self.logger.info("Started experiment timing")
    
    def log_training_step(
        self,
        iter_num: int,
        loss: float,
        lr: float,
        log_interval: int,
        batch_size: int,
        context_length: int
    ):
        """Log training step metrics.
        
        Args:
            iter_num: Current training iteration
            loss: Training loss value
            lr: Current learning rate
            log_interval: Steps between logging
            batch_size: Training batch size
            context_length: Sequence length
        """
        current_time = time.time()
        elapsed_time = current_time - self.training_start_time
        step_time = current_time - self.step_start_time
        steps_per_sec = log_interval / step_time if step_time > 0 else 0
        
        # Console logging
        self.logger.info(
            f"Iter {iter_num}: loss={loss:.4f}, lr={lr:.2e}, "
            f"elapsed={elapsed_time:.1f}s, steps/sec={steps_per_sec:.2f}"
        )
        
        # W&B logging
        if self.wandb_logger:
            self.wandb_logger.log({
                "train/loss": loss,
                "train/lr": lr,
                "train/iter": iter_num,
                "timing/elapsed_time_seconds": elapsed_time,
                "timing/elapsed_time_minutes": elapsed_time / 60,
                "timing/steps_per_second": steps_per_sec,
                "timing/tokens_per_second": steps_per_sec * batch_size * context_length
            })
        
        # Reset step timer
        self.step_start_time = current_time
    
    def log_evaluation(
        self,
        iter_num: int,
        train_loss: float,
        val_loss: float,
        eval_time: float
    ):
        """Log evaluation metrics.
        
        Args:
            iter_num: Current training iteration
            train_loss: Training loss from evaluation
            val_loss: Validation loss
            eval_time: Time spent on evaluation
        """
        current_time = time.time()
        elapsed_time = current_time - self.training_start_time
        
        # Console logging
        self.logger.info(
            f"Iter {iter_num}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"eval_time={eval_time:.1f}s, total_elapsed={elapsed_time/60:.1f}min"
        )
        
        # W&B logging
        if self.wandb_logger:
            self.wandb_logger.log({
                "eval/train_loss": train_loss,
                "eval/val_loss": val_loss,
                "eval/iter": iter_num,
                "timing/eval_time_seconds": eval_time,
                "timing/total_elapsed_minutes": elapsed_time / 60,
                "timing/total_elapsed_hours": elapsed_time / 3600
            })
    
    def log_final_results(
        self,
        train_loss: float,
        val_loss: float
    ):
        """Log final training results.
        
        Args:
            train_loss: Final training loss
            val_loss: Final validation loss
        """
        final_time = time.time()
        total_training_time = final_time - self.training_start_time
        
        # Console logging
        self.logger.info(f"Final: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        self.logger.info(
            f"Total training time: {total_training_time/3600:.2f} hours "
            f"({total_training_time/60:.1f} minutes)"
        )
        
        # W&B logging
        if self.wandb_logger:
            self.wandb_logger.log({
                "final/train_loss": train_loss,
                "final/val_loss": val_loss,
                "timing/total_training_time_hours": total_training_time / 3600,
                "timing/total_training_time_minutes": total_training_time / 60
            })
    
    def log_checkpoint_save(self, iter_num: int, checkpoint_path: str):
        """Log checkpoint saving.
        
        Args:
            iter_num: Current iteration
            checkpoint_path: Path where checkpoint was saved
        """
        self.logger.info(f"Saved checkpoint at iter {iter_num}: {checkpoint_path}")
        
        if self.wandb_logger:
            self.wandb_logger.log({
                "checkpoints/saved_at_iter": iter_num
            })
    
    def log_model_info(self, n_params: int, model_config: Dict[str, Any]):
        """Log model information.
        
        Args:
            n_params: Number of model parameters
            model_config: Model configuration dictionary
        """
        self.logger.info(f"Model has {n_params:,} parameters")
        
        if self.wandb_logger:
            self.wandb_logger.log({
                "model/n_parameters": n_params,
                "model/n_parameters_millions": n_params / 1e6
            })
            # Log model config
            for key, value in model_config.items():
                self.wandb_logger.log({f"model/{key}": value})
    
    def finish(self):
        """Clean up logging resources."""
        if self.wandb_logger:
            self.wandb_logger.finish()
            self.logger.info("Finished W&B logging session")