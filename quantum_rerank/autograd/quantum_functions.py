"""
Custom PyTorch autograd functions for quantum operations.

This module implements custom autograd Function classes that enable gradient computation
through quantum circuits using parameter-shift rules and other gradient estimation methods.
"""

import torch
import torch.autograd as autograd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import time

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class QuantumSimilarityFunction(autograd.Function):
    """
    Custom autograd function for quantum similarity computation.
    
    Enables gradient computation through quantum fidelity operations using
    parameter-shift rules for differentiable quantum machine learning.
    """
    
    @staticmethod
    def forward(ctx, embeddings1: torch.Tensor, embeddings2: torch.Tensor, 
                quantum_params1: torch.Tensor, quantum_params2: torch.Tensor,
                quantum_circuit_func: Callable) -> torch.Tensor:
        """
        Forward pass: compute quantum similarity/fidelity between two states.
        
        Args:
            ctx: Autograd context for saving tensors
            embeddings1, embeddings2: Classical embedding vectors
            quantum_params1, quantum_params2: Quantum circuit parameters
            quantum_circuit_func: Quantum circuit function for fidelity computation
            
        Returns:
            Quantum similarity/fidelity value(s)
        """
        # Save tensors for backward pass
        ctx.save_for_backward(embeddings1, embeddings2, quantum_params1, quantum_params2)
        
        # Save non-tensor data
        ctx.quantum_circuit_func = quantum_circuit_func
        ctx.device = embeddings1.device
        
        # Convert to numpy for quantum circuit execution
        emb1_np = embeddings1.detach().cpu().numpy()
        emb2_np = embeddings2.detach().cpu().numpy()
        params1_np = quantum_params1.detach().cpu().numpy()
        params2_np = quantum_params2.detach().cpu().numpy()
        
        # Execute quantum circuit
        try:
            if emb1_np.ndim == 1:  # Single pair
                fidelity = quantum_circuit_func(params1_np, params2_np, emb1_np, emb2_np)
                if isinstance(fidelity, (list, np.ndarray)):
                    fidelity = fidelity[0] if len(fidelity) > 0 else 0.0
            else:  # Batch processing
                batch_size = emb1_np.shape[0]
                fidelities = []
                for i in range(batch_size):
                    fid = quantum_circuit_func(
                        params1_np[i], params2_np[i], 
                        emb1_np[i], emb2_np[i]
                    )
                    if isinstance(fid, (list, np.ndarray)):
                        fid = fid[0] if len(fid) > 0 else 0.0
                    fidelities.append(fid)
                fidelity = np.array(fidelities)
                
        except Exception as e:
            logger.error(f"Quantum circuit execution failed: {e}")
            # Fallback to classical cosine similarity
            fidelity = torch.cosine_similarity(embeddings1, embeddings2, dim=-1).cpu().numpy()
        
        # Convert back to tensor
        result = torch.tensor(fidelity, dtype=embeddings1.dtype, device=embeddings1.device)
        if result.ndim == 0:
            result = result.unsqueeze(0)
            
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass: compute gradients using parameter-shift rule.
        
        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient from upstream operations
            
        Returns:
            Gradients for all inputs
        """
        embeddings1, embeddings2, quantum_params1, quantum_params2 = ctx.saved_tensors
        quantum_circuit_func = ctx.quantum_circuit_func
        device = ctx.device
        
        # Initialize gradients
        grad_embeddings1 = torch.zeros_like(embeddings1)
        grad_embeddings2 = torch.zeros_like(embeddings2)
        grad_quantum_params1 = torch.zeros_like(quantum_params1)
        grad_quantum_params2 = torch.zeros_like(quantum_params2)
        
        # Parameter-shift rule for quantum parameters
        shift = np.pi / 2
        
        try:
            # Gradient computation for quantum_params1
            for i in range(quantum_params1.shape[-1]):
                # Forward shift
                params1_plus = quantum_params1.clone()
                params1_plus[..., i] += shift
                
                # Backward shift  
                params1_minus = quantum_params1.clone()
                params1_minus[..., i] -= shift
                
                # Compute shifted fidelities
                fidelity_plus = QuantumSimilarityFunction.apply(
                    embeddings1, embeddings2, params1_plus, quantum_params2, quantum_circuit_func
                )
                fidelity_minus = QuantumSimilarityFunction.apply(
                    embeddings1, embeddings2, params1_minus, quantum_params2, quantum_circuit_func
                )
                
                # Parameter-shift gradient
                param_grad = (fidelity_plus - fidelity_minus) / 2
                grad_quantum_params1[..., i] = (grad_output * param_grad).sum()
            
            # Gradient computation for quantum_params2 (similar process)
            for i in range(quantum_params2.shape[-1]):
                params2_plus = quantum_params2.clone()
                params2_plus[..., i] += shift
                
                params2_minus = quantum_params2.clone()
                params2_minus[..., i] -= shift
                
                fidelity_plus = QuantumSimilarityFunction.apply(
                    embeddings1, embeddings2, quantum_params1, params2_plus, quantum_circuit_func
                )
                fidelity_minus = QuantumSimilarityFunction.apply(
                    embeddings1, embeddings2, quantum_params1, params2_minus, quantum_circuit_func
                )
                
                param_grad = (fidelity_plus - fidelity_minus) / 2
                grad_quantum_params2[..., i] = (grad_output * param_grad).sum()
                
        except Exception as e:
            logger.warning(f"Parameter-shift gradient computation failed: {e}")
            # Use finite differences as fallback
            grad_quantum_params1, grad_quantum_params2 = _finite_difference_gradients(
                embeddings1, embeddings2, quantum_params1, quantum_params2,
                quantum_circuit_func, grad_output
            )
        
        # For embeddings, we assume they're not directly differentiable through quantum circuit
        # but may have gradients from the embedding encoding process
        # This would be implemented based on specific embedding encoding method
        
        return grad_embeddings1, grad_embeddings2, grad_quantum_params1, grad_quantum_params2, None


class QuantumParameterFunction(autograd.Function):
    """
    Custom autograd function for quantum parameter prediction from embeddings.
    
    Bridges classical MLP parameter prediction with quantum circuit execution.
    """
    
    @staticmethod
    def forward(ctx, embeddings: torch.Tensor, mlp_output: torch.Tensor,
                scaling_factor: float = np.pi) -> torch.Tensor:
        """
        Forward pass: scale MLP output to quantum parameter range.
        
        Args:
            ctx: Autograd context
            embeddings: Input embeddings
            mlp_output: Raw MLP output (typically in [-1, 1])
            scaling_factor: Scaling factor for quantum parameters
            
        Returns:
            Scaled quantum parameters
        """
        ctx.save_for_backward(embeddings, mlp_output)
        ctx.scaling_factor = scaling_factor
        
        # Scale to quantum parameter range [-π, π]
        quantum_params = mlp_output * scaling_factor
        
        return quantum_params
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass: propagate gradients through parameter scaling.
        
        Args:
            ctx: Autograd context
            grad_output: Gradient from downstream operations
            
        Returns:
            Gradients for inputs
        """
        embeddings, mlp_output = ctx.saved_tensors
        scaling_factor = ctx.scaling_factor
        
        # Gradient w.r.t. MLP output
        grad_mlp_output = grad_output * scaling_factor
        
        # No gradient w.r.t. embeddings at this stage (handled by MLP)
        grad_embeddings = torch.zeros_like(embeddings)
        
        return grad_embeddings, grad_mlp_output, None


class QuantumFidelityFunction(autograd.Function):
    """
    Custom autograd function specifically for quantum fidelity computation.
    
    Implements SWAP test fidelity with proper gradient computation.
    """
    
    @staticmethod 
    def forward(ctx, state_params1: torch.Tensor, state_params2: torch.Tensor,
                embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                num_qubits: int = 4) -> torch.Tensor:
        """
        Forward pass: compute quantum fidelity via SWAP test.
        
        Args:
            ctx: Autograd context
            state_params1, state_params2: Quantum state parameters
            embeddings1, embeddings2: Classical embeddings to encode
            num_qubits: Number of qubits for quantum states
            
        Returns:
            Fidelity values
        """
        ctx.save_for_backward(state_params1, state_params2, embeddings1, embeddings2)
        ctx.num_qubits = num_qubits
        
        # Import quantum simulation here to avoid circular imports
        try:
            from ..quantum.circuits.swap_test import SwapTest
            swap_test = SwapTest(num_qubits)
            
            # Execute SWAP test
            if state_params1.ndim == 1:  # Single computation
                fidelity = swap_test.compute_fidelity(
                    state_params1.detach().cpu().numpy(),
                    state_params2.detach().cpu().numpy(),
                    embeddings1.detach().cpu().numpy(),
                    embeddings2.detach().cpu().numpy()
                )
                fidelity = [fidelity] if not isinstance(fidelity, (list, np.ndarray)) else fidelity
            else:  # Batch computation
                batch_size = state_params1.shape[0]
                fidelities = []
                for i in range(batch_size):
                    fid = swap_test.compute_fidelity(
                        state_params1[i].detach().cpu().numpy(),
                        state_params2[i].detach().cpu().numpy(),
                        embeddings1[i].detach().cpu().numpy(),
                        embeddings2[i].detach().cpu().numpy()
                    )
                    fidelities.append(fid if isinstance(fid, (int, float)) else fid[0])
                fidelity = fidelities
                
        except ImportError:
            logger.warning("SWAP test implementation not available, using fallback")
            # Fallback to classical similarity
            fidelity = torch.cosine_similarity(embeddings1, embeddings2, dim=-1).cpu().numpy()
            
        # Convert to tensor
        result = torch.tensor(fidelity, dtype=state_params1.dtype, device=state_params1.device)
        if result.ndim == 0:
            result = result.unsqueeze(0)
            
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass: compute gradients for fidelity w.r.t. quantum parameters.
        
        Args:
            ctx: Autograd context
            grad_output: Gradient from upstream
            
        Returns:
            Gradients for inputs
        """
        state_params1, state_params2, embeddings1, embeddings2 = ctx.saved_tensors
        num_qubits = ctx.num_qubits
        
        grad_params1 = torch.zeros_like(state_params1)
        grad_params2 = torch.zeros_like(state_params2)
        
        # Parameter-shift rule for fidelity gradients
        shift = np.pi / 2
        
        try:
            # Gradients for state_params1
            for i in range(state_params1.shape[-1]):
                params1_plus = state_params1.clone()
                params1_plus[..., i] += shift
                
                params1_minus = state_params1.clone()
                params1_minus[..., i] -= shift
                
                fidelity_plus = QuantumFidelityFunction.apply(
                    params1_plus, state_params2, embeddings1, embeddings2, num_qubits
                )
                fidelity_minus = QuantumFidelityFunction.apply(
                    params1_minus, state_params2, embeddings1, embeddings2, num_qubits
                )
                
                param_grad = (fidelity_plus - fidelity_minus) / 2
                grad_params1[..., i] = (grad_output * param_grad).sum()
            
            # Gradients for state_params2
            for i in range(state_params2.shape[-1]):
                params2_plus = state_params2.clone()
                params2_plus[..., i] += shift
                
                params2_minus = state_params2.clone()
                params2_minus[..., i] -= shift
                
                fidelity_plus = QuantumFidelityFunction.apply(
                    state_params1, params2_plus, embeddings1, embeddings2, num_qubits
                )
                fidelity_minus = QuantumFidelityFunction.apply(
                    state_params1, params2_minus, embeddings1, embeddings2, num_qubits
                )
                
                param_grad = (fidelity_plus - fidelity_minus) / 2
                grad_params2[..., i] = (grad_output * param_grad).sum()
                
        except Exception as e:
            logger.warning(f"Fidelity gradient computation failed: {e}")
            # Fallback to finite differences
            grad_params1, grad_params2 = _finite_difference_fidelity_gradients(
                state_params1, state_params2, embeddings1, embeddings2,
                num_qubits, grad_output
            )
        
        # No gradients for embeddings in this formulation
        grad_embeddings1 = torch.zeros_like(embeddings1)
        grad_embeddings2 = torch.zeros_like(embeddings2)
        
        return grad_params1, grad_params2, grad_embeddings1, grad_embeddings2, None


def _finite_difference_gradients(embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                                quantum_params1: torch.Tensor, quantum_params2: torch.Tensor,
                                quantum_circuit_func: Callable, grad_output: torch.Tensor,
                                h: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback finite difference gradient computation.
    
    Args:
        embeddings1, embeddings2: Embedding tensors
        quantum_params1, quantum_params2: Quantum parameter tensors
        quantum_circuit_func: Quantum circuit function
        grad_output: Upstream gradients
        h: Finite difference step size
        
    Returns:
        Gradients for quantum parameters
    """
    grad_params1 = torch.zeros_like(quantum_params1)
    grad_params2 = torch.zeros_like(quantum_params2)
    
    # Finite differences for quantum_params1
    for i in range(quantum_params1.shape[-1]):
        params1_plus = quantum_params1.clone()
        params1_plus[..., i] += h
        
        params1_minus = quantum_params1.clone() 
        params1_minus[..., i] -= h
        
        fidelity_plus = QuantumSimilarityFunction.apply(
            embeddings1, embeddings2, params1_plus, quantum_params2, quantum_circuit_func
        )
        fidelity_minus = QuantumSimilarityFunction.apply(
            embeddings1, embeddings2, params1_minus, quantum_params2, quantum_circuit_func
        )
        
        finite_diff_grad = (fidelity_plus - fidelity_minus) / (2 * h)
        grad_params1[..., i] = (grad_output * finite_diff_grad).sum()
    
    # Finite differences for quantum_params2
    for i in range(quantum_params2.shape[-1]):
        params2_plus = quantum_params2.clone()
        params2_plus[..., i] += h
        
        params2_minus = quantum_params2.clone()
        params2_minus[..., i] -= h
        
        fidelity_plus = QuantumSimilarityFunction.apply(
            embeddings1, embeddings2, quantum_params1, params2_plus, quantum_circuit_func
        )
        fidelity_minus = QuantumSimilarityFunction.apply(
            embeddings1, embeddings2, quantum_params1, params2_minus, quantum_circuit_func
        )
        
        finite_diff_grad = (fidelity_plus - fidelity_minus) / (2 * h)
        grad_params2[..., i] = (grad_output * finite_diff_grad).sum()
    
    return grad_params1, grad_params2


def _finite_difference_fidelity_gradients(state_params1: torch.Tensor, state_params2: torch.Tensor,
                                         embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                                         num_qubits: int, grad_output: torch.Tensor,
                                         h: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback finite difference gradient computation for fidelity.
    
    Args:
        state_params1, state_params2: State parameter tensors
        embeddings1, embeddings2: Embedding tensors
        num_qubits: Number of qubits
        grad_output: Upstream gradients
        h: Finite difference step size
        
    Returns:
        Gradients for state parameters
    """
    grad_params1 = torch.zeros_like(state_params1)
    grad_params2 = torch.zeros_like(state_params2)
    
    # Finite differences for state_params1
    for i in range(state_params1.shape[-1]):
        params1_plus = state_params1.clone()
        params1_plus[..., i] += h
        
        params1_minus = state_params1.clone()
        params1_minus[..., i] -= h
        
        fidelity_plus = QuantumFidelityFunction.apply(
            params1_plus, state_params2, embeddings1, embeddings2, num_qubits
        )
        fidelity_minus = QuantumFidelityFunction.apply(
            params1_minus, state_params2, embeddings1, embeddings2, num_qubits
        )
        
        finite_diff_grad = (fidelity_plus - fidelity_minus) / (2 * h)
        grad_params1[..., i] = (grad_output * finite_diff_grad).sum()
    
    # Finite differences for state_params2
    for i in range(state_params2.shape[-1]):
        params2_plus = state_params2.clone()
        params2_plus[..., i] += h
        
        params2_minus = state_params2.clone()
        params2_minus[..., i] -= h
        
        fidelity_plus = QuantumFidelityFunction.apply(
            state_params1, params2_plus, embeddings1, embeddings2, num_qubits
        )
        fidelity_minus = QuantumFidelityFunction.apply(
            state_params1, params2_minus, embeddings1, embeddings2, num_qubits
        )
        
        finite_diff_grad = (fidelity_plus - fidelity_minus) / (2 * h)
        grad_params2[..., i] = (grad_output * finite_diff_grad).sum()
    
    return grad_params1, grad_params2


# Convenience functions for easier usage
def quantum_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                      quantum_params1: torch.Tensor, quantum_params2: torch.Tensor,
                      quantum_circuit_func: Callable) -> torch.Tensor:
    """
    Compute quantum similarity with automatic gradient computation.
    
    Args:
        embeddings1, embeddings2: Classical embeddings
        quantum_params1, quantum_params2: Quantum circuit parameters
        quantum_circuit_func: Quantum circuit function
        
    Returns:
        Quantum similarity values with gradient support
    """
    return QuantumSimilarityFunction.apply(
        embeddings1, embeddings2, quantum_params1, quantum_params2, quantum_circuit_func
    )


def quantum_fidelity(state_params1: torch.Tensor, state_params2: torch.Tensor,
                     embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                     num_qubits: int = 4) -> torch.Tensor:
    """
    Compute quantum fidelity with automatic gradient computation.
    
    Args:
        state_params1, state_params2: Quantum state parameters
        embeddings1, embeddings2: Classical embeddings
        num_qubits: Number of qubits for quantum states
        
    Returns:
        Quantum fidelity values with gradient support
    """
    return QuantumFidelityFunction.apply(
        state_params1, state_params2, embeddings1, embeddings2, num_qubits
    )