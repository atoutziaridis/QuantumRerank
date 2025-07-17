"""
Differential Privacy Module

Provides differential privacy mechanisms for quantum-inspired RAG systems,
including noise addition, privacy budget management, and privacy-preserving
similarity computation.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PrivacyMechanism(Enum):
    """Types of differential privacy mechanisms."""
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5  # Privacy parameter for (ε,δ)-DP
    mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN
    sensitivity: float = 1.0  # Global sensitivity
    clipping_threshold: float = 1.0  # Gradient clipping threshold
    enable_privacy_accounting: bool = True
    
    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta < 0 or self.delta >= 1:
            raise ValueError("Delta must be in [0, 1)")
        if self.sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")


class PrivacyAccountant:
    """Tracks privacy budget usage across operations."""
    
    def __init__(self, total_epsilon: float, total_delta: float):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.used_epsilon = 0.0
        self.used_delta = 0.0
        self.operations_log = []
    
    def spend_privacy_budget(self, epsilon: float, delta: float, operation_name: str) -> bool:
        """Spend privacy budget for an operation."""
        if self.used_epsilon + epsilon > self.total_epsilon:
            logger.warning(f"Insufficient privacy budget for {operation_name}")
            return False
        
        if self.used_delta + delta > self.total_delta:
            logger.warning(f"Insufficient delta budget for {operation_name}")
            return False
        
        self.used_epsilon += epsilon
        self.used_delta += delta
        self.operations_log.append({
            "operation": operation_name,
            "epsilon": epsilon,
            "delta": delta,
            "cumulative_epsilon": self.used_epsilon,
            "cumulative_delta": self.used_delta
        })
        
        logger.info(f"Privacy budget spent on {operation_name}: ε={epsilon:.4f}, δ={delta:.8f}")
        return True
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        return (
            self.total_epsilon - self.used_epsilon,
            self.total_delta - self.used_delta
        )
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get detailed budget status."""
        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "used_epsilon": self.used_epsilon,
            "used_delta": self.used_delta,
            "remaining_epsilon": self.total_epsilon - self.used_epsilon,
            "remaining_delta": self.total_delta - self.used_delta,
            "operations_count": len(self.operations_log)
        }


class DifferentialPrivacy:
    """Differential privacy implementation for quantum-inspired RAG."""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.accountant = PrivacyAccountant(config.epsilon, config.delta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Differential Privacy initialized: ε={config.epsilon}, δ={config.delta}")
    
    def add_noise(self, tensor: torch.Tensor, operation_name: str = "add_noise") -> torch.Tensor:
        """Add differential privacy noise to a tensor."""
        if not self.config.enable_privacy_accounting:
            return self._add_noise_direct(tensor)
        
        # Check privacy budget
        if not self.accountant.spend_privacy_budget(
            self.config.epsilon / 10,  # Use 1/10 of budget per operation
            self.config.delta / 10,
            operation_name
        ):
            logger.warning("Privacy budget exhausted, returning original tensor")
            return tensor
        
        return self._add_noise_direct(tensor)
    
    def _add_noise_direct(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add noise directly without budget checking."""
        if self.config.mechanism == PrivacyMechanism.LAPLACE:
            return self._add_laplace_noise(tensor)
        elif self.config.mechanism == PrivacyMechanism.GAUSSIAN:
            return self._add_gaussian_noise(tensor)
        elif self.config.mechanism == PrivacyMechanism.EXPONENTIAL:
            return self._add_exponential_noise(tensor)
        else:
            raise ValueError(f"Unknown privacy mechanism: {self.config.mechanism}")
    
    def _add_laplace_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Laplace noise for differential privacy."""
        # Scale parameter for Laplace distribution
        scale = self.config.sensitivity / self.config.epsilon
        
        # Generate Laplace noise
        noise = torch.distributions.Laplace(0, scale).sample(tensor.shape).to(tensor.device)
        
        return tensor + noise
    
    def _add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise for differential privacy."""
        # Standard deviation for Gaussian mechanism
        sigma = self.config.sensitivity * np.sqrt(2 * np.log(1.25 / self.config.delta)) / self.config.epsilon
        
        # Generate Gaussian noise
        noise = torch.normal(0, sigma, tensor.shape).to(tensor.device)
        
        return tensor + noise
    
    def _add_exponential_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add exponential noise for differential privacy."""
        # Rate parameter for exponential distribution
        rate = self.config.epsilon / self.config.sensitivity
        
        # Generate exponential noise (centered around 0)
        noise_positive = torch.distributions.Exponential(rate).sample(tensor.shape).to(tensor.device)
        noise_negative = torch.distributions.Exponential(rate).sample(tensor.shape).to(tensor.device)
        noise = noise_positive - noise_negative
        
        return tensor + noise
    
    def clip_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Clip gradients to bound sensitivity."""
        norm = torch.norm(gradients)
        if norm > self.config.clipping_threshold:
            gradients = gradients * (self.config.clipping_threshold / norm)
        
        return gradients
    
    def private_similarity_computation(self, query: torch.Tensor, 
                                     documents: torch.Tensor) -> torch.Tensor:
        """Compute similarity with differential privacy."""
        # Clip inputs to bound sensitivity
        query_clipped = torch.clamp(query, -self.config.clipping_threshold, self.config.clipping_threshold)
        docs_clipped = torch.clamp(documents, -self.config.clipping_threshold, self.config.clipping_threshold)
        
        # Compute similarity
        similarities = torch.cosine_similarity(
            query_clipped.unsqueeze(0), docs_clipped, dim=1
        )
        
        # Add noise to similarity scores
        private_similarities = self.add_noise(similarities, "similarity_computation")
        
        return private_similarities
    
    def private_aggregation(self, values: torch.Tensor, 
                          aggregation_type: str = "mean") -> torch.Tensor:
        """Perform private aggregation of values."""
        if aggregation_type == "mean":
            result = torch.mean(values)
        elif aggregation_type == "sum":
            result = torch.sum(values)
        elif aggregation_type == "max":
            result = torch.max(values)
        elif aggregation_type == "min":
            result = torch.min(values)
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
        
        # Add noise to aggregated result
        private_result = self.add_noise(result.unsqueeze(0), f"aggregation_{aggregation_type}")
        
        return private_result.squeeze(0)
    
    def exponential_mechanism(self, candidates: List[Any], 
                            utility_function: callable,
                            context: Any = None) -> Any:
        """Select candidate using exponential mechanism."""
        # Compute utility scores
        utilities = []
        for candidate in candidates:
            utility = utility_function(candidate, context) if context else utility_function(candidate)
            utilities.append(utility)
        
        utilities = torch.tensor(utilities, dtype=torch.float32)
        
        # Apply exponential mechanism
        scaled_utilities = utilities * self.config.epsilon / (2 * self.config.sensitivity)
        probabilities = torch.softmax(scaled_utilities, dim=0)
        
        # Sample from distribution
        selected_idx = torch.multinomial(probabilities, 1).item()
        
        # Log privacy budget usage
        if self.config.enable_privacy_accounting:
            self.accountant.spend_privacy_budget(
                self.config.epsilon / 10,
                0.0,  # Exponential mechanism doesn't use delta
                "exponential_mechanism"
            )
        
        return candidates[selected_idx]
    
    def private_top_k_selection(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Select top-k items with differential privacy."""
        # Add noise to scores
        private_scores = self.add_noise(scores, "top_k_selection")
        
        # Select top-k based on noisy scores
        top_k_indices = torch.topk(private_scores, k).indices
        
        return top_k_indices
    
    def get_privacy_guarantees(self) -> Dict[str, Any]:
        """Get current privacy guarantees."""
        remaining_epsilon, remaining_delta = self.accountant.get_remaining_budget()
        
        return {
            "epsilon": self.config.epsilon,
            "delta": self.config.delta,
            "mechanism": self.config.mechanism.value,
            "sensitivity": self.config.sensitivity,
            "remaining_epsilon": remaining_epsilon,
            "remaining_delta": remaining_delta,
            "privacy_exhausted": remaining_epsilon <= 0,
            "budget_status": self.accountant.get_budget_status()
        }
    
    def reset_privacy_budget(self):
        """Reset privacy budget to initial values."""
        self.accountant = PrivacyAccountant(self.config.epsilon, self.config.delta)
        logger.info("Privacy budget reset")
    
    def analyze_privacy_loss(self) -> Dict[str, Any]:
        """Analyze privacy loss from operations."""
        operations = self.accountant.operations_log
        
        if not operations:
            return {"status": "no_operations"}
        
        # Calculate privacy loss statistics
        epsilon_values = [op["epsilon"] for op in operations]
        delta_values = [op["delta"] for op in operations]
        
        analysis = {
            "total_operations": len(operations),
            "total_epsilon_spent": sum(epsilon_values),
            "total_delta_spent": sum(delta_values),
            "average_epsilon_per_operation": np.mean(epsilon_values),
            "average_delta_per_operation": np.mean(delta_values),
            "max_epsilon_single_operation": max(epsilon_values),
            "max_delta_single_operation": max(delta_values),
            "operations_breakdown": {},
            "privacy_remaining": self.accountant.get_remaining_budget()
        }
        
        # Breakdown by operation type
        for op in operations:
            op_name = op["operation"]
            if op_name not in analysis["operations_breakdown"]:
                analysis["operations_breakdown"][op_name] = {
                    "count": 0,
                    "total_epsilon": 0,
                    "total_delta": 0
                }
            
            analysis["operations_breakdown"][op_name]["count"] += 1
            analysis["operations_breakdown"][op_name]["total_epsilon"] += op["epsilon"]
            analysis["operations_breakdown"][op_name]["total_delta"] += op["delta"]
        
        return analysis


# Utility functions
def calculate_noise_scale(epsilon: float, delta: float, sensitivity: float, 
                         mechanism: PrivacyMechanism) -> float:
    """Calculate noise scale for given privacy parameters."""
    if mechanism == PrivacyMechanism.LAPLACE:
        return sensitivity / epsilon
    elif mechanism == PrivacyMechanism.GAUSSIAN:
        return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    else:
        raise ValueError(f"Noise scale calculation not implemented for {mechanism}")


def estimate_privacy_loss(operations: List[Dict[str, float]]) -> Tuple[float, float]:
    """Estimate total privacy loss from a list of operations."""
    total_epsilon = sum(op.get("epsilon", 0) for op in operations)
    total_delta = sum(op.get("delta", 0) for op in operations)
    
    return total_epsilon, total_delta


def compose_privacy_guarantees(epsilon1: float, delta1: float, 
                              epsilon2: float, delta2: float) -> Tuple[float, float]:
    """Compose privacy guarantees using advanced composition."""
    # Simple composition (conservative bound)
    composed_epsilon = epsilon1 + epsilon2
    composed_delta = delta1 + delta2
    
    return composed_epsilon, composed_delta