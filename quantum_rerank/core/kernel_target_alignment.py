"""
Kernel Target Alignment (KTA) implementation for quantum kernel optimization.

Implements data-driven quantum kernel optimization using KTA as the primary
objective function for circuit parameter tuning and performance evaluation.

Based on:
- "Data-Driven Quantum Kernel Development" documentation
- KTA optimization for quantum machine learning
- Problem-specific quantum feature map adaptation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.optimize import minimize
import time

logger = logging.getLogger(__name__)

@dataclass
class KTAConfig:
    """Configuration for Kernel Target Alignment optimization."""
    optimization_method: str = "L-BFGS-B"  # Optimization algorithm
    max_iterations: int = 100  # Maximum optimization iterations
    tolerance: float = 1e-6  # Convergence tolerance
    regularization: float = 1e-8  # Regularization parameter
    kernel_normalization: bool = True  # Normalize kernel matrix
    target_encoding: str = "binary"  # Target encoding method: binary, categorical
    parallel_computation: bool = False  # Parallel kernel computation
    cache_kernel_matrices: bool = True  # Cache computed kernels


class KernelTargetAlignment:
    """
    Kernel Target Alignment optimizer for quantum kernels.
    
    Implements data-driven optimization of quantum kernel parameters
    using KTA as the objective function.
    """
    
    def __init__(self, config: KTAConfig = None):
        self.config = config or KTAConfig()
        self.kernel_cache = {} if self.config.cache_kernel_matrices else None
        self.optimization_history = []
        
    def compute_kta(self, kernel_matrix: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Kernel Target Alignment between kernel matrix and target labels.
        
        KTA(K, y) = <K, yy^T>_F / sqrt(<K, K>_F * <yy^T, yy^T>_F)
        
        Args:
            kernel_matrix: Kernel matrix (n_samples, n_samples)
            labels: Target labels (n_samples,)
            
        Returns:
            KTA score between 0 and 1
        """
        # Handle different label encodings
        if self.config.target_encoding == "binary":
            # Convert to binary labels (-1, 1)
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                y_binary = np.where(labels == unique_labels[0], -1, 1)
            else:
                # Multi-class: use one-vs-rest for first class
                y_binary = np.where(labels == unique_labels[0], 1, -1)
            target_vector = y_binary
        else:
            target_vector = labels
        
        # Create target matrix
        target_matrix = np.outer(target_vector, target_vector)
        
        # Add regularization to prevent numerical instability
        if self.config.regularization > 0:
            kernel_matrix = kernel_matrix + self.config.regularization * np.eye(kernel_matrix.shape[0])
        
        # Normalize kernel matrix if specified
        if self.config.kernel_normalization:
            kernel_matrix = self._normalize_kernel_matrix(kernel_matrix)
        
        # Compute Frobenius inner products
        kk_inner = np.trace(kernel_matrix @ kernel_matrix.T)
        yy_inner = np.trace(target_matrix @ target_matrix.T)
        ky_inner = np.trace(kernel_matrix @ target_matrix.T)
        
        # Avoid division by zero
        denominator = np.sqrt(kk_inner * yy_inner)
        if denominator < 1e-12:
            logger.warning("Near-zero denominator in KTA computation")
            return 0.0
        
        # Calculate KTA
        kta = ky_inner / denominator
        
        # KTA should be between -1 and 1, but we often want positive values
        return abs(kta)
    
    def _normalize_kernel_matrix(self, kernel_matrix: np.ndarray) -> np.ndarray:
        """Normalize kernel matrix to have unit diagonal."""
        diag_sqrt = np.sqrt(np.diag(kernel_matrix))
        # Avoid division by zero
        diag_sqrt = np.where(diag_sqrt < 1e-12, 1.0, diag_sqrt)
        
        normalized = kernel_matrix / np.outer(diag_sqrt, diag_sqrt)
        return normalized
    
    def optimize_quantum_kernel_parameters(self, 
                                         quantum_kernel_func: Callable,
                                         training_data: np.ndarray,
                                         labels: np.ndarray,
                                         initial_parameters: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize quantum kernel parameters using KTA as objective function.
        
        Args:
            quantum_kernel_func: Function that computes kernel matrix given parameters
            training_data: Training data (n_samples, n_features)
            labels: Training labels (n_samples,)
            initial_parameters: Initial parameter values
            
        Returns:
            Tuple of (optimized_parameters, optimization_info)
        """
        logger.info(f"Starting KTA optimization with {len(training_data)} samples")
        
        # Clear optimization history
        self.optimization_history = []
        
        def objective_function(params):
            """Objective function: negative KTA (for minimization)."""
            try:
                # Compute kernel matrix with current parameters
                kernel_matrix = quantum_kernel_func(training_data, params)
                
                # Compute KTA
                kta_score = self.compute_kta(kernel_matrix, labels)
                
                # Store optimization step
                self.optimization_history.append({
                    'params': params.copy(),
                    'kta_score': kta_score,
                    'timestamp': time.time()
                })
                
                # Return negative KTA for minimization
                objective_value = -kta_score
                
                logger.debug(f"KTA: {kta_score:.6f}, Objective: {objective_value:.6f}")
                return objective_value
                
            except Exception as e:
                logger.warning(f"Error in objective function: {e}")
                return 1.0  # Return high value on error
        
        # Optimization
        logger.info(f"Starting optimization with {self.config.optimization_method}")
        start_time = time.time()
        
        try:
            result = minimize(
                objective_function,
                initial_parameters,
                method=self.config.optimization_method,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.tolerance,
                    'disp': False
                }
            )
            
            optimization_time = time.time() - start_time
            
            # Get final KTA score
            final_kernel = quantum_kernel_func(training_data, result.x)
            final_kta = self.compute_kta(final_kernel, labels)
            
            optimization_info = {
                'success': result.success,
                'message': result.message,
                'iterations': result.nit,
                'function_evaluations': result.nfev,
                'initial_kta': -result.fun if len(self.optimization_history) > 0 else 0.0,
                'final_kta': final_kta,
                'optimization_time': optimization_time,
                'optimization_history': self.optimization_history
            }
            
            logger.info(f"Optimization completed: KTA improved from "
                       f"{optimization_info['initial_kta']:.6f} to {final_kta:.6f}")
            
            return result.x, optimization_info
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return initial_parameters, {
                'success': False,
                'message': str(e),
                'final_kta': 0.0,
                'optimization_time': time.time() - start_time
            }
    
    def evaluate_kernel_quality(self, kernel_matrix: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive evaluation of kernel quality metrics.
        
        Args:
            kernel_matrix: Kernel matrix to evaluate
            labels: Target labels
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Kernel Target Alignment
        metrics['kta'] = self.compute_kta(kernel_matrix, labels)
        
        # Kernel alignment with ideal kernel
        ideal_kernel = self._create_ideal_kernel(labels)
        metrics['ideal_alignment'] = self._kernel_alignment(kernel_matrix, ideal_kernel)
        
        # Spectral properties
        try:
            eigenvalues = np.linalg.eigvals(kernel_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Filter near-zero eigenvalues
            
            if len(eigenvalues) > 0:
                metrics['effective_dimension'] = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
                metrics['condition_number'] = np.max(eigenvalues) / np.min(eigenvalues)
                metrics['spectral_gap'] = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if len(eigenvalues) > 1 else 1.0
            else:
                metrics['effective_dimension'] = 0.0
                metrics['condition_number'] = float('inf')
                metrics['spectral_gap'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error computing spectral properties: {e}")
            metrics['effective_dimension'] = 0.0
            metrics['condition_number'] = float('inf')
            metrics['spectral_gap'] = 0.0
        
        # Kernel concentration
        metrics['kernel_concentration'] = self._compute_kernel_concentration(kernel_matrix)
        
        # Frobenius norm
        metrics['frobenius_norm'] = np.linalg.norm(kernel_matrix, 'fro')
        
        return metrics
    
    def _create_ideal_kernel(self, labels: np.ndarray) -> np.ndarray:
        """Create ideal kernel matrix based on labels."""
        # Ideal kernel: 1 for same class, 0 for different class
        return (labels[:, np.newaxis] == labels[np.newaxis, :]).astype(float)
    
    def _kernel_alignment(self, K1: np.ndarray, K2: np.ndarray) -> float:
        """Compute alignment between two kernel matrices."""
        numerator = np.trace(K1 @ K2.T)
        denominator = np.sqrt(np.trace(K1 @ K1.T) * np.trace(K2 @ K2.T))
        
        if denominator < 1e-12:
            return 0.0
        
        return numerator / denominator
    
    def _compute_kernel_concentration(self, kernel_matrix: np.ndarray) -> float:
        """
        Compute kernel concentration measure.
        
        High concentration indicates vanishing similarity problem.
        """
        # Compute variance of off-diagonal elements
        n = kernel_matrix.shape[0]
        if n <= 1:
            return 0.0
        
        # Get off-diagonal elements
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = kernel_matrix[mask]
        
        # Concentration measure: inverse of variance (higher = more concentrated)
        variance = np.var(off_diagonal)
        if variance < 1e-12:
            return float('inf')  # Highly concentrated
        
        return 1.0 / variance
    
    def compare_kernels(self, 
                       kernel_matrices: Dict[str, np.ndarray], 
                       labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple kernel matrices using various quality metrics.
        
        Args:
            kernel_matrices: Dictionary of kernel name -> kernel matrix
            labels: Target labels
            
        Returns:
            Dictionary of kernel evaluations
        """
        results = {}
        
        for kernel_name, kernel_matrix in kernel_matrices.items():
            logger.info(f"Evaluating kernel: {kernel_name}")
            results[kernel_name] = self.evaluate_kernel_quality(kernel_matrix, labels)
        
        return results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization process."""
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        kta_scores = [step['kta_score'] for step in self.optimization_history]
        
        return {
            "total_iterations": len(self.optimization_history),
            "initial_kta": kta_scores[0],
            "final_kta": kta_scores[-1],
            "best_kta": max(kta_scores),
            "improvement": kta_scores[-1] - kta_scores[0],
            "convergence_rate": self._compute_convergence_rate(kta_scores)
        }
    
    def _compute_convergence_rate(self, kta_scores: List[float]) -> float:
        """Compute convergence rate of optimization."""
        if len(kta_scores) < 2:
            return 0.0
        
        # Simple convergence rate: average improvement per iteration
        total_improvement = kta_scores[-1] - kta_scores[0]
        iterations = len(kta_scores) - 1
        
        return total_improvement / iterations if iterations > 0 else 0.0


class AdaptiveKTAOptimizer:
    """
    Adaptive KTA optimizer that adjusts optimization strategy based on performance.
    """
    
    def __init__(self, base_kta: KernelTargetAlignment):
        self.base_kta = base_kta
        self.optimization_strategies = [
            {"method": "L-BFGS-B", "max_iter": 100},
            {"method": "SLSQP", "max_iter": 150},
            {"method": "TNC", "max_iter": 200}
        ]
    
    def adaptive_optimize(self, 
                         quantum_kernel_func: Callable,
                         training_data: np.ndarray,
                         labels: np.ndarray,
                         initial_parameters: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Try multiple optimization strategies and return the best result.
        """
        best_params = initial_parameters
        best_kta = 0.0
        best_info = {}
        
        for i, strategy in enumerate(self.optimization_strategies):
            logger.info(f"Trying optimization strategy {i+1}: {strategy['method']}")
            
            # Update configuration
            self.base_kta.config.optimization_method = strategy["method"]
            self.base_kta.config.max_iterations = strategy["max_iter"]
            
            try:
                params, info = self.base_kta.optimize_quantum_kernel_parameters(
                    quantum_kernel_func, training_data, labels, initial_parameters
                )
                
                if info.get('final_kta', 0.0) > best_kta:
                    best_params = params
                    best_kta = info['final_kta']
                    best_info = info
                    best_info['best_strategy'] = strategy['method']
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy['method']} failed: {e}")
                continue
        
        logger.info(f"Best optimization strategy: {best_info.get('best_strategy', 'None')} "
                   f"with KTA: {best_kta:.6f}")
        
        return best_params, best_info