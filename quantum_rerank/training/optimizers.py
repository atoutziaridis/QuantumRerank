"""
Optimization strategies for hybrid quantum-classical training.

This module provides quantum parameter optimization, classical optimizer integration,
and hybrid gradient combination strategies for the QuantumRerank training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

from ..utils import get_logger


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    gradient_clipping: float = 1.0
    lr_schedule: str = "cosine"  # "constant", "linear", "cosine", "exponential"
    warmup_steps: int = 1000
    total_steps: int = 10000


class QuantumGradientComputer:
    """
    Quantum gradient computation using parameter-shift rule.
    
    This class computes gradients for quantum circuit parameters using
    the parameter-shift rule, which is the standard method for gradient
    computation in variational quantum algorithms.
    """
    
    def __init__(self, shift_value: float = np.pi / 2):
        self.shift_value = shift_value
        self.logger = get_logger(__name__)
    
    def compute_gradients(
        self,
        quantum_circuit_func: Callable,
        parameters: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute gradients using parameter-shift rule.
        
        Args:
            quantum_circuit_func: Function that executes quantum circuit
            parameters: Quantum circuit parameters
            *args, **kwargs: Additional arguments for quantum circuit
            
        Returns:
            Computed gradients
        """
        gradients = torch.zeros_like(parameters)
        
        for i, param in enumerate(parameters):
            # Positive shift
            params_plus = parameters.clone()
            params_plus[i] += self.shift_value
            result_plus = quantum_circuit_func(params_plus, *args, **kwargs)
            
            # Negative shift
            params_minus = parameters.clone()
            params_minus[i] -= self.shift_value
            result_minus = quantum_circuit_func(params_minus, *args, **kwargs)
            
            # Gradient using parameter-shift rule
            gradient = (result_plus - result_minus) / (2 * np.sin(self.shift_value))
            gradients[i] = gradient
        
        return gradients


class QuantumOptimizer:
    """
    Quantum parameter optimizer with constraints and specialized update rules.
    
    This optimizer is designed specifically for quantum circuit parameters,
    incorporating constraints and quantum-aware optimization strategies.
    """
    
    def __init__(
        self,
        parameters: torch.Tensor,
        learning_rate: float = 0.01,
        parameter_bounds: Optional[Tuple[float, float]] = None,
        constraint_strength: float = 0.1
    ):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.parameter_bounds = parameter_bounds or (-2 * np.pi, 2 * np.pi)
        self.constraint_strength = constraint_strength
        self.logger = get_logger(__name__)
        
        # Optimizer state
        self.step_count = 0
        self.momentum_buffer = torch.zeros_like(parameters)
        self.velocity = torch.zeros_like(parameters)
        
        # Gradient computer
        self.gradient_computer = QuantumGradientComputer()
    
    def step(
        self,
        quantum_circuit_func: Callable,
        loss_func: Callable,
        *args,
        **kwargs
    ) -> float:
        """
        Perform optimization step.
        
        Args:
            quantum_circuit_func: Quantum circuit execution function
            loss_func: Loss function to minimize
            *args, **kwargs: Additional arguments
            
        Returns:
            Current loss value
        """
        self.step_count += 1
        
        # Compute current loss
        current_loss = loss_func(quantum_circuit_func(self.parameters, *args, **kwargs))
        
        # Compute gradients using parameter-shift rule
        gradients = self._compute_quantum_gradients(
            quantum_circuit_func, loss_func, *args, **kwargs
        )
        
        # Apply parameter constraints
        constrained_gradients = self._apply_constraints(gradients)
        
        # Update parameters using momentum
        self.momentum_buffer = (
            0.9 * self.momentum_buffer + self.learning_rate * constrained_gradients
        )
        
        self.parameters.data -= self.momentum_buffer
        
        # Enforce parameter bounds
        self._enforce_bounds()
        
        return current_loss.item() if isinstance(current_loss, torch.Tensor) else current_loss
    
    def _compute_quantum_gradients(
        self,
        quantum_circuit_func: Callable,
        loss_func: Callable,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Compute quantum gradients using parameter-shift rule."""
        gradients = torch.zeros_like(self.parameters)
        
        for i in range(len(self.parameters)):
            # Positive shift
            params_plus = self.parameters.clone()
            params_plus[i] += np.pi / 2
            loss_plus = loss_func(quantum_circuit_func(params_plus, *args, **kwargs))
            
            # Negative shift
            params_minus = self.parameters.clone()
            params_minus[i] -= np.pi / 2
            loss_minus = loss_func(quantum_circuit_func(params_minus, *args, **kwargs))
            
            # Gradient
            gradient = (loss_plus - loss_minus) / 2.0
            gradients[i] = gradient
        
        return gradients
    
    def _apply_constraints(self, gradients: torch.Tensor) -> torch.Tensor:
        """Apply quantum parameter constraints."""
        # Add penalty for parameters approaching bounds
        lower_bound, upper_bound = self.parameter_bounds
        
        # Penalty for parameters near lower bound
        lower_penalty = torch.where(
            self.parameters < lower_bound + 0.1,
            -self.constraint_strength * (self.parameters - lower_bound - 0.1),
            torch.zeros_like(self.parameters)
        )
        
        # Penalty for parameters near upper bound
        upper_penalty = torch.where(
            self.parameters > upper_bound - 0.1,
            -self.constraint_strength * (self.parameters - upper_bound + 0.1),
            torch.zeros_like(self.parameters)
        )
        
        return gradients + lower_penalty + upper_penalty
    
    def _enforce_bounds(self) -> None:
        """Enforce parameter bounds."""
        lower_bound, upper_bound = self.parameter_bounds
        self.parameters.data = torch.clamp(
            self.parameters.data, lower_bound, upper_bound
        )
    
    def get_parameter_statistics(self) -> Dict[str, float]:
        """Get statistics about current parameters."""
        return {
            "mean": self.parameters.mean().item(),
            "std": self.parameters.std().item(),
            "min": self.parameters.min().item(),
            "max": self.parameters.max().item(),
            "gradient_norm": self.momentum_buffer.norm().item()
        }


class HybridOptimizer:
    """
    Hybrid optimizer combining quantum and classical optimization strategies.
    
    This optimizer coordinates between quantum parameter optimization and
    classical neural network training with proper gradient combination.
    """
    
    def __init__(
        self,
        quantum_parameters: torch.Tensor,
        classical_parameters: List[torch.Tensor],
        quantum_lr: float = 0.01,
        classical_lr: float = 0.001,
        quantum_weight: float = 0.5,
        classical_weight: float = 0.5
    ):
        self.quantum_optimizer = QuantumOptimizer(quantum_parameters, quantum_lr)
        
        # Classical optimizer (Adam)
        self.classical_optimizer = torch.optim.Adam(
            classical_parameters, lr=classical_lr
        )
        
        self.quantum_weight = quantum_weight
        self.classical_weight = classical_weight
        self.logger = get_logger(__name__)
        
        # Training statistics
        self.quantum_losses = []
        self.classical_losses = []
        self.combined_losses = []
    
    def step(
        self,
        quantum_loss: torch.Tensor,
        classical_loss: torch.Tensor,
        quantum_circuit_func: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Perform hybrid optimization step.
        
        Args:
            quantum_loss: Loss from quantum components
            classical_loss: Loss from classical components
            quantum_circuit_func: Quantum circuit function for gradient computation
            
        Returns:
            Dictionary of loss components
        """
        # Combine losses
        total_loss = (
            self.quantum_weight * quantum_loss + 
            self.classical_weight * classical_loss
        )
        
        # Classical optimization step
        self.classical_optimizer.zero_grad()
        classical_loss.backward(retain_graph=True)
        self.classical_optimizer.step()
        
        # Quantum optimization step (if circuit function provided)
        if quantum_circuit_func is not None:
            quantum_loss_value = self.quantum_optimizer.step(
                quantum_circuit_func,
                lambda x: quantum_loss
            )
        else:
            quantum_loss_value = quantum_loss.item()
        
        # Record losses
        self.quantum_losses.append(quantum_loss_value)
        self.classical_losses.append(classical_loss.item())
        self.combined_losses.append(total_loss.item())
        
        return {
            "quantum_loss": quantum_loss_value,
            "classical_loss": classical_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def adjust_weights(self, quantum_performance: float, classical_performance: float) -> None:
        """Dynamically adjust quantum vs classical optimization weights."""
        total_performance = quantum_performance + classical_performance
        
        if total_performance > 0:
            self.quantum_weight = quantum_performance / total_performance
            self.classical_weight = classical_performance / total_performance
        
        self.logger.debug(
            f"Adjusted optimization weights - Quantum: {self.quantum_weight:.3f}, "
            f"Classical: {self.classical_weight:.3f}"
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            "quantum_stats": self.quantum_optimizer.get_parameter_statistics(),
            "classical_lr": self.classical_optimizer.param_groups[0]["lr"],
            "quantum_weight": self.quantum_weight,
            "classical_weight": self.classical_weight,
            "loss_history": {
                "quantum": self.quantum_losses[-10:],  # Last 10 steps
                "classical": self.classical_losses[-10:],
                "combined": self.combined_losses[-10:]
            }
        }


class LearningRateScheduler:
    """Learning rate scheduling for hybrid training."""
    
    def __init__(
        self,
        optimizer: Union[torch.optim.Optimizer, HybridOptimizer],
        schedule_type: str = "cosine",
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        min_lr_factor: float = 0.01
    ):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_factor = min_lr_factor
        self.step_count = 0
        
        # Store initial learning rates
        if isinstance(optimizer, torch.optim.Optimizer):
            self.initial_lrs = [group["lr"] for group in optimizer.param_groups]
        else:
            self.initial_quantum_lr = optimizer.quantum_optimizer.learning_rate
            self.initial_classical_lr = optimizer.classical_optimizer.param_groups[0]["lr"]
    
    def step(self) -> None:
        """Update learning rates according to schedule."""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr_factor = self.step_count / self.warmup_steps
        else:
            # Main scheduling phase
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            if self.schedule_type == "cosine":
                lr_factor = (
                    self.min_lr_factor + 
                    0.5 * (1 - self.min_lr_factor) * (1 + np.cos(np.pi * progress))
                )
            elif self.schedule_type == "linear":
                lr_factor = 1.0 - progress * (1 - self.min_lr_factor)
            elif self.schedule_type == "exponential":
                lr_factor = self.min_lr_factor ** progress
            else:
                lr_factor = 1.0  # Constant
        
        # Apply learning rate updates
        if isinstance(self.optimizer, torch.optim.Optimizer):
            for param_group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
                param_group["lr"] = initial_lr * lr_factor
        else:
            # Hybrid optimizer
            self.optimizer.quantum_optimizer.learning_rate = self.initial_quantum_lr * lr_factor
            for param_group in self.optimizer.classical_optimizer.param_groups:
                param_group["lr"] = self.initial_classical_lr * lr_factor
    
    def get_current_lr(self) -> Union[float, Dict[str, float]]:
        """Get current learning rate(s)."""
        if isinstance(self.optimizer, torch.optim.Optimizer):
            return self.optimizer.param_groups[0]["lr"]
        else:
            return {
                "quantum": self.optimizer.quantum_optimizer.learning_rate,
                "classical": self.optimizer.classical_optimizer.param_groups[0]["lr"]
            }


def create_quantum_optimizer(
    parameters: torch.Tensor,
    config: OptimizerConfig
) -> QuantumOptimizer:
    """Create quantum optimizer with configuration."""
    return QuantumOptimizer(
        parameters=parameters,
        learning_rate=config.learning_rate,
        parameter_bounds=(-2 * np.pi, 2 * np.pi),
        constraint_strength=0.1
    )


def create_hybrid_optimizer(
    quantum_parameters: torch.Tensor,
    classical_parameters: List[torch.Tensor],
    config: OptimizerConfig
) -> HybridOptimizer:
    """Create hybrid optimizer with configuration."""
    return HybridOptimizer(
        quantum_parameters=quantum_parameters,
        classical_parameters=classical_parameters,
        quantum_lr=config.learning_rate * 0.1,  # Lower LR for quantum
        classical_lr=config.learning_rate,
        quantum_weight=0.5,
        classical_weight=0.5
    )


def create_lr_scheduler(
    optimizer: Union[torch.optim.Optimizer, HybridOptimizer],
    config: OptimizerConfig
) -> LearningRateScheduler:
    """Create learning rate scheduler with configuration."""
    return LearningRateScheduler(
        optimizer=optimizer,
        schedule_type=config.lr_schedule,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        min_lr_factor=0.01
    )


class GradientAnalyzer:
    """Analyzer for gradient flow and optimization health."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.gradient_norms = []
        self.parameter_changes = []
        self.logger = get_logger(__name__)
    
    def analyze_gradients(
        self,
        model: nn.Module,
        loss: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze gradient properties for optimization health."""
        # Compute gradients
        model.zero_grad()
        loss.backward(retain_graph=True)
        
        # Collect gradient statistics
        total_norm = 0.0
        total_params = 0
        max_grad = 0.0
        min_grad = float('inf')
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                total_params += param.numel()
                
                max_grad = max(max_grad, param.grad.data.max().item())
                min_grad = min(min_grad, param.grad.data.min().item())
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Maintain sliding window
        if len(self.gradient_norms) > self.window_size:
            self.gradient_norms = self.gradient_norms[-self.window_size:]
        
        # Compute analysis metrics
        analysis = {
            "gradient_norm": total_norm,
            "max_gradient": max_grad,
            "min_gradient": min_grad,
            "avg_gradient_norm": np.mean(self.gradient_norms),
            "gradient_norm_std": np.std(self.gradient_norms),
            "total_parameters": total_params
        }
        
        # Detect potential issues
        if total_norm > 10.0:
            self.logger.warning("Large gradient norm detected: gradient explosion risk")
        elif total_norm < 1e-6:
            self.logger.warning("Very small gradient norm: vanishing gradient risk")
        
        return analysis


__all__ = [
    "OptimizerConfig",
    "QuantumGradientComputer",
    "QuantumOptimizer",
    "HybridOptimizer",
    "LearningRateScheduler",
    "GradientAnalyzer",
    "create_quantum_optimizer",
    "create_hybrid_optimizer",
    "create_lr_scheduler"
]