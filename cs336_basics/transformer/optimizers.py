"""Optimizers for transformer training."""

import math
import torch
from collections.abc import Callable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """Construct AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: Learning rate
            betas: Coefficients used for computing running averages of gradient and its square
            eps: Term added to the denominator for numerical stability
            weight_decay: Weight decay coefficient
        """

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)


    @torch.no_grad()                     # ← turn autograd off inside step
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            # Extract hyperparameters
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Get parameter state
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["t"] = 1 # Initialize iteration number
                    state["m"] = torch.zeros_like(p)  # First moment estimate (m_t)
                    state["v"] = torch.zeros_like(p)  # Second moment estimate (v_t)
                
                t, m, v = state["t"], state["m"], state["v"]
                grad = p.grad
                
                # moments (in-place updates)
                m.mul_(beta1).add_(grad, alpha=1 - beta1) # Update 1t moment: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # Update 2nd moment: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

                # learning rate
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                
                # Update weight
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                # Apply weight decay
                p.add_(p, alpha=-lr * weight_decay)

                # Update state
                state["t"] += 1
                
        return loss