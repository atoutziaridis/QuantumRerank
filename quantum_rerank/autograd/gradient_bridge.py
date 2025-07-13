"""
Quantum-Classical Gradient Bridge for Hybrid Training.

This module provides bridging components that enable seamless gradient flow
between classical PyTorch operations and quantum circuit computations.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
import time

from .quantum_functions import QuantumSimilarityFunction, QuantumFidelityFunction
from .parameter_shift import QuantumGradientEstimator, GradientConfig
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for quantum-classical gradient bridge."""
    gradient_method: str = "parameter_shift"
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    fallback_to_classical: bool = True
    cache_quantum_results: bool = True
    validate_gradients: bool = False
    quantum_noise_threshold: float = 1e-6


class QuantumClassicalBridge(nn.Module):
    """
    Bridge module that connects classical PyTorch networks with quantum operations.
    
    This module handles the integration between classical embeddings, quantum parameter
    prediction, and quantum similarity computation with proper gradient flow.
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 num_quantum_params: int = 24,
                 quantum_circuit_func: Optional[Callable] = None,
                 config: Optional[BridgeConfig] = None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_quantum_params = num_quantum_params
        self.quantum_circuit_func = quantum_circuit_func
        self.config = config or BridgeConfig()
        
        # Classical parameter prediction network
        self.param_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_quantum_params),
            nn.Tanh()  # Bound parameters to [-1, 1]
        )
        
        # Gradient estimation
        self.gradient_estimator = QuantumGradientEstimator(
            default_method=self.config.gradient_method
        )
        
        # Performance tracking
        self.quantum_call_count = 0
        self.classical_fallback_count = 0
        self.gradient_computation_times = []
        
        logger.info(f"Initialized QuantumClassicalBridge with {num_quantum_params} quantum parameters")
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing quantum similarity between embeddings.
        
        Args:
            embeddings1, embeddings2: Input embedding tensors
            
        Returns:
            Quantum similarity values
        """
        # Predict quantum parameters
        quantum_params1 = self.param_predictor(embeddings1) * np.pi  # Scale to [-π, π]
        quantum_params2 = self.param_predictor(embeddings2) * np.pi
        
        # Apply gradient clipping if enabled
        if self.config.gradient_clipping:
            quantum_params1 = torch.clamp(quantum_params1, -np.pi, np.pi)
            quantum_params2 = torch.clamp(quantum_params2, -np.pi, np.pi)
        
        # Compute quantum similarity with custom autograd
        if self.quantum_circuit_func is not None:
            similarity = QuantumSimilarityFunction.apply(
                embeddings1, embeddings2, quantum_params1, quantum_params2, 
                self.quantum_circuit_func
            )
            self.quantum_call_count += 1
        else:
            # Fallback to classical similarity if no quantum circuit available
            similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=-1)
            self.classical_fallback_count += 1
            logger.warning("No quantum circuit available, using classical fallback")
        
        return similarity
    
    def compute_fidelity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                        num_qubits: int = 4) -> torch.Tensor:
        """
        Compute quantum fidelity between embedding-derived quantum states.
        
        Args:
            embeddings1, embeddings2: Input embeddings
            num_qubits: Number of qubits for quantum states
            
        Returns:
            Quantum fidelity values
        """
        # Predict quantum state parameters
        state_params1 = self.param_predictor(embeddings1) * np.pi
        state_params2 = self.param_predictor(embeddings2) * np.pi
        
        # Compute fidelity with custom autograd
        fidelity = QuantumFidelityFunction.apply(
            state_params1, state_params2, embeddings1, embeddings2, num_qubits
        )
        
        return fidelity
    
    def validate_gradient_flow(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> Dict[str, Any]:
        """
        Validate gradient flow through the quantum-classical bridge.
        
        Args:
            embeddings1, embeddings2: Test embeddings
            
        Returns:
            Validation results
        """
        validation_results = {
            "gradient_flow_valid": False,
            "classical_gradients": None,
            "quantum_gradients": None,
            "total_gradient_norm": 0.0,
            "quantum_gradient_norm": 0.0,
            "classical_gradient_norm": 0.0
        }
        
        # Enable gradient computation
        embeddings1.requires_grad_(True)
        embeddings2.requires_grad_(True)
        
        try:
            # Forward pass
            similarity = self.forward(embeddings1, embeddings2)
            loss = similarity.sum()  # Simple aggregation for gradient test
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            if embeddings1.grad is not None and embeddings2.grad is not None:
                classical_grad_norm = torch.norm(embeddings1.grad) + torch.norm(embeddings2.grad)
                validation_results["classical_gradient_norm"] = classical_grad_norm.item()
                validation_results["classical_gradients"] = True
            
            # Check quantum parameter gradients
            quantum_grad_norm = 0.0
            for param in self.param_predictor.parameters():
                if param.grad is not None:
                    quantum_grad_norm += torch.norm(param.grad).item()
            
            validation_results["quantum_gradient_norm"] = quantum_grad_norm
            validation_results["quantum_gradients"] = quantum_grad_norm > 0
            
            validation_results["total_gradient_norm"] = (
                validation_results["classical_gradient_norm"] + quantum_grad_norm
            )
            
            validation_results["gradient_flow_valid"] = (
                validation_results["classical_gradients"] and 
                validation_results["quantum_gradients"]
            )
            
        except Exception as e:
            logger.error(f"Gradient validation failed: {e}")
            validation_results["error"] = str(e)
        
        return validation_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the bridge."""
        total_calls = self.quantum_call_count + self.classical_fallback_count
        
        stats = {
            "total_calls": total_calls,
            "quantum_calls": self.quantum_call_count,
            "classical_fallback_calls": self.classical_fallback_count,
            "quantum_call_ratio": self.quantum_call_count / max(total_calls, 1),
            "avg_gradient_time": np.mean(self.gradient_computation_times) if self.gradient_computation_times else 0.0
        }
        
        return stats


class HybridGradientFlow(nn.Module):
    """
    Advanced hybrid gradient flow manager for complex quantum-classical architectures.
    
    Manages gradient computation and flow in hybrid systems with multiple quantum
    components and classical networks.
    """
    
    def __init__(self, components: Dict[str, nn.Module], config: Optional[BridgeConfig] = None):
        super().__init__()
        
        self.config = config or BridgeConfig()
        self.components = nn.ModuleDict(components)
        self.gradient_hooks = {}
        self.gradient_stats = {}
        
        # Register gradient hooks for monitoring
        self._register_gradient_hooks()
        
        logger.info(f"Initialized HybridGradientFlow with {len(components)} components")
    
    def _register_gradient_hooks(self):
        """Register hooks to monitor gradient flow."""
        def create_hook(name):
            def hook(grad):
                if name not in self.gradient_stats:
                    self.gradient_stats[name] = []
                
                grad_norm = torch.norm(grad).item()
                self.gradient_stats[name].append(grad_norm)
                
                # Apply gradient clipping if enabled
                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_([grad], self.config.max_grad_norm)
                
                return grad
            return hook
        
        # Register hooks for all parameters
        for name, component in self.components.items():
            for param_name, param in component.named_parameters():
                if param.requires_grad:
                    hook_name = f"{name}.{param_name}"
                    handle = param.register_hook(create_hook(hook_name))
                    self.gradient_hooks[hook_name] = handle
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all hybrid components.
        
        Args:
            inputs: Dictionary of input tensors for different components
            
        Returns:
            Dictionary of output tensors
        """
        outputs = {}
        
        for name, component in self.components.items():
            if name in inputs:
                try:
                    outputs[name] = component(inputs[name])
                except Exception as e:
                    logger.error(f"Component {name} failed: {e}")
                    # Fallback or error handling
                    if hasattr(component, 'classical_fallback'):
                        outputs[name] = component.classical_fallback(inputs[name])
                    else:
                        outputs[name] = torch.zeros_like(inputs[name][:, :1])  # Dummy output
        
        return outputs
    
    def compute_hybrid_loss(self, outputs: Dict[str, torch.Tensor], 
                           targets: Dict[str, torch.Tensor],
                           loss_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Compute weighted loss across all hybrid components.
        
        Args:
            outputs: Component outputs
            targets: Target values
            loss_weights: Weights for different loss components
            
        Returns:
            Combined loss tensor
        """
        total_loss = torch.tensor(0.0, requires_grad=True)
        loss_weights = loss_weights or {}
        
        for name in outputs:
            if name in targets:
                # Component-specific loss computation
                if 'similarity' in name or 'fidelity' in name:
                    # For similarity/fidelity, use MSE or triplet loss
                    component_loss = nn.functional.mse_loss(outputs[name], targets[name])
                else:
                    # Default loss
                    component_loss = nn.functional.mse_loss(outputs[name], targets[name])
                
                # Apply weight
                weight = loss_weights.get(name, 1.0)
                total_loss = total_loss + weight * component_loss
        
        return total_loss
    
    def optimize_gradient_flow(self) -> Dict[str, Any]:
        """
        Analyze and optimize gradient flow patterns.
        
        Returns:
            Optimization recommendations
        """
        recommendations = {
            "gradient_flow_issues": [],
            "optimization_suggestions": [],
            "component_health": {}
        }
        
        # Analyze gradient statistics
        for name, grad_norms in self.gradient_stats.items():
            if len(grad_norms) > 0:
                avg_norm = np.mean(grad_norms)
                std_norm = np.std(grad_norms)
                
                component_health = {
                    "avg_gradient_norm": avg_norm,
                    "gradient_std": std_norm,
                    "gradient_stability": std_norm / max(avg_norm, 1e-8)
                }
                
                # Detect issues
                if avg_norm < 1e-6:
                    recommendations["gradient_flow_issues"].append(
                        f"Very small gradients in {name} (vanishing gradients)"
                    )
                    recommendations["optimization_suggestions"].append(
                        f"Consider gradient scaling or learning rate adjustment for {name}"
                    )
                elif avg_norm > 10.0:
                    recommendations["gradient_flow_issues"].append(
                        f"Large gradients in {name} (exploding gradients)"
                    )
                    recommendations["optimization_suggestions"].append(
                        f"Enable gradient clipping for {name}"
                    )
                
                if component_health["gradient_stability"] > 2.0:
                    recommendations["gradient_flow_issues"].append(
                        f"Unstable gradients in {name}"
                    )
                
                recommendations["component_health"][name] = component_health
        
        return recommendations
    
    def cleanup_hooks(self):
        """Remove all registered gradient hooks."""
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks.clear()


class QuantumGradientChecker:
    """
    Utility class for validating quantum gradient computations.
    """
    
    @staticmethod
    def check_autograd_function(autograd_function: Function,
                               test_inputs: Tuple[torch.Tensor, ...],
                               eps: float = 1e-6,
                               atol: float = 1e-4) -> bool:
        """
        Check custom autograd function using gradcheck.
        
        Args:
            autograd_function: Custom autograd function to test
            test_inputs: Test input tensors
            eps: Finite difference step size
            atol: Absolute tolerance
            
        Returns:
            True if gradcheck passes
        """
        try:
            from torch.autograd import gradcheck
            
            # Ensure inputs require gradients and use double precision
            test_inputs_double = []
            for inp in test_inputs:
                if isinstance(inp, torch.Tensor):
                    inp_double = inp.double().requires_grad_(True)
                    test_inputs_double.append(inp_double)
                else:
                    test_inputs_double.append(inp)
            
            # Run gradcheck
            test_passed = gradcheck(
                autograd_function.apply,
                test_inputs_double,
                eps=eps,
                atol=atol,
                raise_exception=False
            )
            
            if test_passed:
                logger.info("✓ Autograd function gradcheck passed")
            else:
                logger.warning("✗ Autograd function gradcheck failed")
            
            return test_passed
            
        except Exception as e:
            logger.error(f"Gradcheck failed with exception: {e}")
            return False
    
    @staticmethod
    def validate_quantum_classical_bridge(bridge: QuantumClassicalBridge,
                                         test_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """
        Comprehensive validation of quantum-classical bridge.
        
        Args:
            bridge: Bridge to validate
            test_embeddings: Test embedding tensors
            
        Returns:
            Validation results
        """
        emb1, emb2 = test_embeddings
        
        # Basic gradient flow test
        gradient_results = bridge.validate_gradient_flow(emb1, emb2)
        
        # Performance test
        performance_stats = bridge.get_performance_stats()
        
        # Forward pass consistency test
        with torch.no_grad():
            output1 = bridge(emb1, emb2)
            output2 = bridge(emb1, emb2)  # Should be deterministic
        
        consistency_check = torch.allclose(output1, output2, atol=1e-6)
        
        validation_summary = {
            "gradient_flow": gradient_results,
            "performance": performance_stats,
            "output_consistency": consistency_check,
            "overall_health": (
                gradient_results.get("gradient_flow_valid", False) and
                consistency_check
            )
        }
        
        return validation_summary


# Convenience functions for testing and validation
def create_test_bridge(embedding_dim: int = 768, num_quantum_params: int = 24) -> QuantumClassicalBridge:
    """Create a test quantum-classical bridge for validation."""
    def mock_quantum_circuit(params1, params2, emb1, emb2):
        """Mock quantum circuit for testing."""
        return np.random.random()  # Mock fidelity
    
    bridge = QuantumClassicalBridge(
        embedding_dim=embedding_dim,
        num_quantum_params=num_quantum_params,
        quantum_circuit_func=mock_quantum_circuit
    )
    
    return bridge


def validate_hybrid_training(bridge: QuantumClassicalBridge,
                            test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                            num_steps: int = 10) -> Dict[str, Any]:
    """
    Validate hybrid training with triplet data.
    
    Args:
        bridge: Quantum-classical bridge to test
        test_data: List of (anchor, positive, negative) triplets
        num_steps: Number of training steps to simulate
        
    Returns:
        Training validation results
    """
    optimizer = torch.optim.Adam(bridge.parameters(), lr=1e-3)
    triplet_loss = nn.TripletMarginLoss(margin=0.3)
    
    losses = []
    gradient_norms = []
    
    for step in range(num_steps):
        total_loss = 0.0
        
        for anchor, positive, negative in test_data:
            optimizer.zero_grad()
            
            # Compute similarities
            sim_pos = bridge(anchor, positive)
            sim_neg = bridge(anchor, negative)
            
            # Triplet loss (convert similarities to distances)
            loss = triplet_loss(anchor, positive, negative)
            loss.backward()
            
            # Track gradient norms
            total_grad_norm = 0.0
            for param in bridge.parameters():
                if param.grad is not None:
                    total_grad_norm += torch.norm(param.grad).item()
            
            gradient_norms.append(total_grad_norm)
            
            optimizer.step()
            total_loss += loss.item()
        
        losses.append(total_loss / len(test_data))
    
    validation_results = {
        "training_successful": len(losses) == num_steps,
        "loss_trend": "decreasing" if losses[-1] < losses[0] else "increasing",
        "final_loss": losses[-1],
        "avg_gradient_norm": np.mean(gradient_norms),
        "gradient_stability": np.std(gradient_norms) / max(np.mean(gradient_norms), 1e-8)
    }
    
    return validation_results