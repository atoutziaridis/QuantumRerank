"""
Parameter-shift rule implementation for quantum gradient computation.

This module provides implementations of the parameter-shift rule and other
gradient estimation methods for quantum circuits integrated with PyTorch.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
import concurrent.futures
import threading

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GradientConfig:
    """Configuration for gradient computation methods."""
    method: str = "parameter_shift"  # "parameter_shift", "finite_diff", "spsa"
    shift_value: float = np.pi / 2  # Shift for parameter-shift rule
    finite_diff_h: float = 1e-6     # Step size for finite differences
    spsa_a: float = 0.1              # SPSA gradient descent parameter
    spsa_c: float = 0.1              # SPSA perturbation magnitude
    parallel: bool = True            # Enable parallel gradient computation
    max_workers: int = 4             # Maximum parallel workers


class ParameterShiftGradient:
    """
    Implementation of parameter-shift rule for quantum gradient computation.
    
    The parameter-shift rule enables exact gradient computation for quantum circuits
    with parameterized gates, avoiding the need for finite differences.
    """
    
    def __init__(self, config: Optional[GradientConfig] = None):
        self.config = config or GradientConfig()
        self.logger = logger
        self._thread_local = threading.local()
        
    def compute_gradient(self, quantum_function: Callable,
                        params: torch.Tensor, 
                        other_args: Tuple = ()) -> torch.Tensor:
        """
        Compute gradient using parameter-shift rule.
        
        Args:
            quantum_function: Quantum function to differentiate
            params: Parameters to compute gradients for
            other_args: Additional arguments to quantum function
            
        Returns:
            Gradient tensor with same shape as params
        """
        if self.config.method == "parameter_shift":
            return self._parameter_shift_gradient(quantum_function, params, other_args)
        elif self.config.method == "finite_diff":
            return self._finite_difference_gradient(quantum_function, params, other_args)
        elif self.config.method == "spsa":
            return self._spsa_gradient(quantum_function, params, other_args)
        else:
            raise ValueError(f"Unknown gradient method: {self.config.method}")
    
    def _parameter_shift_gradient(self, quantum_function: Callable,
                                 params: torch.Tensor,
                                 other_args: Tuple) -> torch.Tensor:
        """
        Compute gradient using parameter-shift rule.
        
        For a parameterized quantum gate with parameter θ, the gradient is:
        ∂⟨Ψ|O|Ψ⟩/∂θ = (1/2)[⟨Ψ(θ+π/2)|O|Ψ(θ+π/2)⟩ - ⟨Ψ(θ-π/2)|O|Ψ(θ-π/2)⟩]
        
        Args:
            quantum_function: Function to differentiate
            params: Parameters tensor
            other_args: Additional function arguments
            
        Returns:
            Parameter-shift gradients
        """
        gradient = torch.zeros_like(params)
        shift = self.config.shift_value
        
        if self.config.parallel and params.numel() > 1:
            # Parallel gradient computation
            gradient = self._parallel_parameter_shift(quantum_function, params, other_args, shift)
        else:
            # Sequential gradient computation
            if params.ndim == 1:
                # Single parameter vector
                for i in range(len(params)):
                    grad_i = self._compute_single_parameter_shift(
                        quantum_function, params, i, shift, other_args
                    )
                    gradient[i] = grad_i
            else:
                # Batch of parameter vectors
                for batch_idx in range(params.shape[0]):
                    for param_idx in range(params.shape[1]):
                        grad_val = self._compute_single_parameter_shift_batch(
                            quantum_function, params, batch_idx, param_idx, shift, other_args
                        )
                        gradient[batch_idx, param_idx] = grad_val
        
        return gradient
    
    def _compute_single_parameter_shift(self, quantum_function: Callable,
                                       params: torch.Tensor, param_idx: int,
                                       shift: float, other_args: Tuple) -> float:
        """Compute parameter-shift gradient for single parameter."""
        # Forward shift: θ + π/2
        params_plus = params.clone()
        params_plus[param_idx] += shift
        
        # Backward shift: θ - π/2
        params_minus = params.clone()
        params_minus[param_idx] -= shift
        
        # Evaluate function at shifted parameters
        try:
            result_plus = quantum_function(params_plus, *other_args)
            result_minus = quantum_function(params_minus, *other_args)
            
            # Extract scalar values
            if isinstance(result_plus, torch.Tensor):
                result_plus = result_plus.item() if result_plus.numel() == 1 else result_plus.mean().item()
            if isinstance(result_minus, torch.Tensor):
                result_minus = result_minus.item() if result_minus.numel() == 1 else result_minus.mean().item()
            
            # Parameter-shift formula
            gradient_val = (result_plus - result_minus) / 2.0
            
        except Exception as e:
            self.logger.warning(f"Parameter-shift computation failed for param {param_idx}: {e}")
            gradient_val = 0.0
        
        return gradient_val
    
    def _compute_single_parameter_shift_batch(self, quantum_function: Callable,
                                            params: torch.Tensor, batch_idx: int,
                                            param_idx: int, shift: float,
                                            other_args: Tuple) -> float:
        """Compute parameter-shift gradient for single parameter in batch."""
        # Forward shift
        params_plus = params.clone()
        params_plus[batch_idx, param_idx] += shift
        
        # Backward shift
        params_minus = params.clone()
        params_minus[batch_idx, param_idx] -= shift
        
        try:
            result_plus = quantum_function(params_plus, *other_args)
            result_minus = quantum_function(params_minus, *other_args)
            
            # Handle batch results
            if isinstance(result_plus, torch.Tensor):
                if result_plus.ndim > 0:
                    result_plus = result_plus[batch_idx].item()
                else:
                    result_plus = result_plus.item()
            
            if isinstance(result_minus, torch.Tensor):
                if result_minus.ndim > 0:
                    result_minus = result_minus[batch_idx].item()
                else:
                    result_minus = result_minus.item()
            
            gradient_val = (result_plus - result_minus) / 2.0
            
        except Exception as e:
            self.logger.warning(f"Batch parameter-shift failed for [{batch_idx},{param_idx}]: {e}")
            gradient_val = 0.0
        
        return gradient_val
    
    def _parallel_parameter_shift(self, quantum_function: Callable,
                                 params: torch.Tensor, other_args: Tuple,
                                 shift: float) -> torch.Tensor:
        """Compute parameter-shift gradients in parallel."""
        gradient = torch.zeros_like(params)
        
        def compute_param_gradient(param_idx):
            """Compute gradient for single parameter index."""
            if params.ndim == 1:
                return param_idx, self._compute_single_parameter_shift(
                    quantum_function, params, param_idx, shift, other_args
                )
            else:
                # For batched parameters, compute all batch gradients for this parameter
                batch_gradients = []
                for batch_idx in range(params.shape[0]):
                    grad_val = self._compute_single_parameter_shift_batch(
                        quantum_function, params, batch_idx, param_idx, shift, other_args
                    )
                    batch_gradients.append(grad_val)
                return param_idx, batch_gradients
        
        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            if params.ndim == 1:
                futures = [executor.submit(compute_param_gradient, i) for i in range(len(params))]
                for future in concurrent.futures.as_completed(futures):
                    param_idx, grad_val = future.result()
                    gradient[param_idx] = grad_val
            else:
                futures = [executor.submit(compute_param_gradient, i) for i in range(params.shape[1])]
                for future in concurrent.futures.as_completed(futures):
                    param_idx, batch_gradients = future.result()
                    for batch_idx, grad_val in enumerate(batch_gradients):
                        gradient[batch_idx, param_idx] = grad_val
        
        return gradient
    
    def _finite_difference_gradient(self, quantum_function: Callable,
                                   params: torch.Tensor,
                                   other_args: Tuple) -> torch.Tensor:
        """
        Compute gradient using finite differences as fallback.
        
        Args:
            quantum_function: Function to differentiate
            params: Parameters tensor
            other_args: Additional function arguments
            
        Returns:
            Finite difference gradients
        """
        gradient = torch.zeros_like(params)
        h = self.config.finite_diff_h
        
        if params.ndim == 1:
            for i in range(len(params)):
                # Forward difference
                params_plus = params.clone()
                params_plus[i] += h
                
                params_minus = params.clone()
                params_minus[i] -= h
                
                try:
                    result_plus = quantum_function(params_plus, *other_args)
                    result_minus = quantum_function(params_minus, *other_args)
                    
                    if isinstance(result_plus, torch.Tensor):
                        result_plus = result_plus.mean().item()
                    if isinstance(result_minus, torch.Tensor):
                        result_minus = result_minus.mean().item()
                    
                    gradient[i] = (result_plus - result_minus) / (2 * h)
                    
                except Exception as e:
                    self.logger.warning(f"Finite difference failed for param {i}: {e}")
                    gradient[i] = 0.0
        else:
            # Batch processing
            for batch_idx in range(params.shape[0]):
                for param_idx in range(params.shape[1]):
                    params_plus = params.clone()
                    params_plus[batch_idx, param_idx] += h
                    
                    params_minus = params.clone()
                    params_minus[batch_idx, param_idx] -= h
                    
                    try:
                        result_plus = quantum_function(params_plus, *other_args)
                        result_minus = quantum_function(params_minus, *other_args)
                        
                        if isinstance(result_plus, torch.Tensor):
                            result_plus = result_plus[batch_idx].item()
                        if isinstance(result_minus, torch.Tensor):
                            result_minus = result_minus[batch_idx].item()
                        
                        gradient[batch_idx, param_idx] = (result_plus - result_minus) / (2 * h)
                        
                    except Exception as e:
                        self.logger.warning(f"Finite difference failed for [{batch_idx},{param_idx}]: {e}")
                        gradient[batch_idx, param_idx] = 0.0
        
        return gradient
    
    def _spsa_gradient(self, quantum_function: Callable,
                      params: torch.Tensor,
                      other_args: Tuple) -> torch.Tensor:
        """
        Compute gradient using Simultaneous Perturbation Stochastic Approximation (SPSA).
        
        SPSA is more efficient for high-dimensional parameter spaces as it requires
        only two function evaluations regardless of parameter count.
        
        Args:
            quantum_function: Function to differentiate
            params: Parameters tensor
            other_args: Additional function arguments
            
        Returns:
            SPSA gradient estimates
        """
        # Generate random perturbation vector (Rademacher distribution)
        if params.ndim == 1:
            delta = torch.randint(0, 2, params.shape, dtype=params.dtype, device=params.device)
            delta = 2 * delta - 1  # Convert {0,1} to {-1,1}
        else:
            delta = torch.randint(0, 2, params.shape, dtype=params.dtype, device=params.device)
            delta = 2 * delta - 1
        
        c_k = self.config.spsa_c  # Perturbation magnitude
        
        # Perturbed parameters
        params_plus = params + c_k * delta
        params_minus = params - c_k * delta
        
        try:
            # Function evaluations
            result_plus = quantum_function(params_plus, *other_args)
            result_minus = quantum_function(params_minus, *other_args)
            
            # Handle tensor results
            if isinstance(result_plus, torch.Tensor):
                result_plus = result_plus.mean().item() if params.ndim == 1 else result_plus.mean(dim=1)
            if isinstance(result_minus, torch.Tensor):
                result_minus = result_minus.mean().item() if params.ndim == 1 else result_minus.mean(dim=1)
            
            # SPSA gradient estimate
            if params.ndim == 1:
                gradient = (result_plus - result_minus) / (2 * c_k) * delta
            else:
                # Batch processing
                gradient = torch.zeros_like(params)
                for batch_idx in range(params.shape[0]):
                    diff = result_plus[batch_idx] - result_minus[batch_idx]
                    gradient[batch_idx] = (diff / (2 * c_k)) * delta[batch_idx]
            
        except Exception as e:
            self.logger.error(f"SPSA gradient computation failed: {e}")
            gradient = torch.zeros_like(params)
        
        return gradient


class QuantumGradientEstimator:
    """
    High-level gradient estimator for quantum operations with automatic method selection.
    """
    
    def __init__(self, default_method: str = "parameter_shift",
                 fallback_method: str = "finite_diff"):
        self.default_method = default_method
        self.fallback_method = fallback_method
        self.parameter_shift = ParameterShiftGradient(GradientConfig(method="parameter_shift"))
        self.finite_diff = ParameterShiftGradient(GradientConfig(method="finite_diff"))
        self.spsa = ParameterShiftGradient(GradientConfig(method="spsa"))
        
    def estimate_gradient(self, quantum_function: Callable,
                         params: torch.Tensor,
                         other_args: Tuple = (),
                         method: Optional[str] = None) -> torch.Tensor:
        """
        Estimate gradient with automatic fallback.
        
        Args:
            quantum_function: Quantum function to differentiate
            params: Parameters to compute gradients for
            other_args: Additional function arguments
            method: Override default gradient method
            
        Returns:
            Estimated gradients
        """
        method = method or self.default_method
        
        try:
            if method == "parameter_shift":
                return self.parameter_shift.compute_gradient(quantum_function, params, other_args)
            elif method == "finite_diff":
                return self.finite_diff.compute_gradient(quantum_function, params, other_args)
            elif method == "spsa":
                return self.spsa.compute_gradient(quantum_function, params, other_args)
            else:
                raise ValueError(f"Unknown gradient method: {method}")
                
        except Exception as e:
            logger.warning(f"Gradient method {method} failed: {e}")
            logger.info(f"Falling back to {self.fallback_method}")
            
            # Fallback gradient computation
            if self.fallback_method == "finite_diff":
                return self.finite_diff.compute_gradient(quantum_function, params, other_args)
            elif self.fallback_method == "spsa":
                return self.spsa.compute_gradient(quantum_function, params, other_args)
            else:
                # Last resort: zero gradients
                logger.error("All gradient methods failed, returning zero gradients")
                return torch.zeros_like(params)
    
    def validate_gradients(self, quantum_function: Callable,
                          params: torch.Tensor,
                          other_args: Tuple = (),
                          tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Validate gradient computation by comparing different methods.
        
        Args:
            quantum_function: Function to validate
            params: Parameters to test
            other_args: Additional function arguments
            tolerance: Tolerance for gradient comparison
            
        Returns:
            Validation results dictionary
        """
        results = {
            "parameter_shift": None,
            "finite_diff": None,
            "spsa": None,
            "max_difference": None,
            "validation_passed": False
        }
        
        try:
            # Compute gradients with different methods
            grad_ps = self.parameter_shift.compute_gradient(quantum_function, params, other_args)
            grad_fd = self.finite_diff.compute_gradient(quantum_function, params, other_args)
            grad_spsa = self.spsa.compute_gradient(quantum_function, params, other_args)
            
            results["parameter_shift"] = grad_ps
            results["finite_diff"] = grad_fd
            results["spsa"] = grad_spsa
            
            # Compare parameter-shift and finite differences (most reliable comparison)
            max_diff = torch.max(torch.abs(grad_ps - grad_fd)).item()
            results["max_difference"] = max_diff
            results["validation_passed"] = max_diff < tolerance
            
            logger.info(f"Gradient validation: max difference = {max_diff:.6f}")
            if results["validation_passed"]:
                logger.info("✓ Gradient validation passed")
            else:
                logger.warning("✗ Gradient validation failed - check implementation")
                
        except Exception as e:
            logger.error(f"Gradient validation failed: {e}")
            results["error"] = str(e)
        
        return results


# Convenience functions for common use cases
def compute_quantum_gradient(quantum_function: Callable, params: torch.Tensor,
                           method: str = "parameter_shift", **kwargs) -> torch.Tensor:
    """
    Convenience function for computing quantum gradients.
    
    Args:
        quantum_function: Quantum function to differentiate
        params: Parameters to compute gradients for
        method: Gradient computation method
        **kwargs: Additional arguments
        
    Returns:
        Computed gradients
    """
    config = GradientConfig(method=method, **kwargs)
    gradient_computer = ParameterShiftGradient(config)
    return gradient_computer.compute_gradient(quantum_function, params)


def validate_quantum_gradients(quantum_function: Callable, params: torch.Tensor,
                              tolerance: float = 1e-3) -> bool:
    """
    Convenience function for validating quantum gradient computations.
    
    Args:
        quantum_function: Function to validate
        params: Parameters to test
        tolerance: Validation tolerance
        
    Returns:
        True if validation passes
    """
    estimator = QuantumGradientEstimator()
    results = estimator.validate_gradients(quantum_function, params, tolerance=tolerance)
    return results.get("validation_passed", False)