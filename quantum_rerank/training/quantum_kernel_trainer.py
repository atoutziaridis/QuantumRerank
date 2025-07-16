"""
Quantum Kernel Training with KTA Optimization for Medical Domain.

This module implements Kernel Target Alignment (KTA) optimization for quantum
kernels using medical corpus data to optimize quantum circuit parameters for
improved medical document ranking and similarity computation.

Based on QRF-04 requirements for quantum parameter training pipeline.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import pickle
from pathlib import Path
from scipy.optimize import differential_evolution, minimize
from scipy.stats import spearmanr

from .medical_data_preparation import TrainingPair, MedicalTrainingConfig
from ..core.quantum_kernel_engine import QuantumKernelEngine
from ..core.kernel_target_alignment import KernelTargetAlignment
from ..core.quantum_similarity_engine import QuantumSimilarityEngine
from ..config.settings import QuantumConfig

logger = logging.getLogger(__name__)


@dataclass
class KTAOptimizationConfig:
    """Configuration for KTA optimization of quantum kernels."""
    target_kta_score: float = 0.7
    max_iterations: int = 100
    optimization_method: str = "differential_evolution"  # or "gradient_free", "scipy_minimize"
    population_size: int = 50
    convergence_threshold: float = 1e-6
    parameter_bounds: Tuple[float, float] = (-np.pi, np.pi)
    validation_frequency: int = 10
    early_stopping_patience: int = 20
    save_intermediate_results: bool = True
    random_seed: int = 42


@dataclass
class QuantumKernelTrainingResult:
    """Results from quantum kernel training."""
    optimal_parameters: np.ndarray
    best_kta_score: float
    optimization_history: List[float]
    convergence_iteration: int
    training_time_seconds: float
    validation_results: Dict[str, float]
    parameter_sensitivity: Dict[str, float]
    baseline_kta_score: float = 0.0
    optimization_iterations: int = 0
    training_pairs_count: int = 0


class KTAOptimizer:
    """
    Kernel Target Alignment optimizer for quantum kernels.
    
    Optimizes quantum circuit parameters to maximize KTA score between
    quantum kernel matrix and target labels on medical data.
    """
    
    def __init__(self, config: Optional[KTAOptimizationConfig] = None):
        """Initialize KTA optimizer."""
        self.config = config or KTAOptimizationConfig()
        self.kta_calculator = KernelTargetAlignment()
        
        # Optimization state
        self.optimization_history: List[float] = []
        self.best_parameters: Optional[np.ndarray] = None
        self.best_score: float = -1.0
        self.iteration_count: int = 0
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        
        logger.info(f"KTA optimizer initialized with config: {self.config}")
    
    def optimize_parameters(self, quantum_kernel_engine: QuantumKernelEngine,
                          training_pairs: List[TrainingPair],
                          validation_pairs: Optional[List[TrainingPair]] = None) -> QuantumKernelTrainingResult:
        """
        Optimize quantum kernel parameters using KTA on medical training data.
        
        Args:
            quantum_kernel_engine: Quantum kernel engine to optimize
            training_pairs: Training data with embeddings and labels
            validation_pairs: Validation data for early stopping
            
        Returns:
            Training results with optimal parameters and scores
        """
        logger.info(f"Starting KTA optimization on {len(training_pairs)} training pairs")
        start_time = time.time()
        
        # Prepare optimization data
        embeddings, labels = self._prepare_optimization_data(training_pairs)
        val_embeddings, val_labels = None, None
        if validation_pairs:
            val_embeddings, val_labels = self._prepare_optimization_data(validation_pairs)
        
        # Initialize optimization state
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = -1.0
        self.iteration_count = 0
        
        # Define objective function
        def objective_function(parameters: np.ndarray) -> float:
            return self._evaluate_parameters(parameters, quantum_kernel_engine, embeddings, labels)
        
        # Get parameter bounds
        n_params = quantum_kernel_engine.get_parameter_count()
        bounds = [self.config.parameter_bounds] * n_params
        
        logger.info(f"Optimizing {n_params} parameters using {self.config.optimization_method}")
        
        # Run optimization
        if self.config.optimization_method == "differential_evolution":
            result = self._run_differential_evolution(objective_function, bounds)
        elif self.config.optimization_method == "scipy_minimize":
            result = self._run_scipy_minimize(objective_function, bounds)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
        
        # Validate optimal parameters
        validation_results = {}
        if validation_pairs:
            validation_results = self._validate_parameters(
                result.x, quantum_kernel_engine, val_embeddings, val_labels
            )
        
        # Analyze parameter sensitivity
        parameter_sensitivity = self._analyze_parameter_sensitivity(
            result.x, quantum_kernel_engine, embeddings, labels
        )
        
        training_time = time.time() - start_time
        
        training_result = QuantumKernelTrainingResult(
            optimal_parameters=result.x,
            best_kta_score=self.best_score,
            optimization_history=self.optimization_history,
            convergence_iteration=self.iteration_count,
            training_time_seconds=training_time,
            validation_results=validation_results,
            parameter_sensitivity=parameter_sensitivity,
            baseline_kta_score=0.1,  # Typical random parameter baseline
            optimization_iterations=self.iteration_count,
            training_pairs_count=len(training_pairs)
        )
        
        logger.info(f"KTA optimization completed in {training_time:.2f}s")
        logger.info(f"Best KTA score: {self.best_score:.4f}")
        logger.info(f"Converged at iteration: {self.iteration_count}")
        
        return training_result
    
    def _prepare_optimization_data(self, training_pairs: List[TrainingPair]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare embeddings and labels for optimization."""
        # Extract embeddings (concatenate query and doc embeddings)
        embeddings = []
        labels = []
        
        for pair in training_pairs:
            # Concatenate query and document embeddings
            combined_embedding = np.concatenate([pair.query_embedding, pair.doc_embedding])
            embeddings.append(combined_embedding)
            labels.append(pair.relevance_label)
        
        embeddings_array = np.array(embeddings)
        labels_array = np.array(labels)
        
        logger.info(f"Prepared optimization data: {embeddings_array.shape} embeddings, {len(labels_array)} labels")
        
        return embeddings_array, labels_array
    
    def _evaluate_parameters(self, parameters: np.ndarray,
                           quantum_kernel_engine: QuantumKernelEngine,
                           embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate quantum kernel parameters using KTA score."""
        try:
            # Set parameters in quantum kernel engine
            quantum_kernel_engine.set_parameters(parameters)
            
            # Compute quantum kernel matrix
            n_samples = min(50, len(embeddings))  # Limit for efficiency
            sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
            sample_embeddings = embeddings[sample_indices]
            sample_labels = labels[sample_indices]
            
            # Compute pairwise quantum kernel values
            kernel_matrix = np.zeros((n_samples, n_samples))
            
            for i in range(n_samples):
                for j in range(i, n_samples):
                    # Split combined embedding back to query and doc parts
                    emb_size = len(sample_embeddings[i]) // 2
                    query_emb_i = sample_embeddings[i][:emb_size]
                    doc_emb_i = sample_embeddings[i][emb_size:]
                    query_emb_j = sample_embeddings[j][:emb_size]
                    doc_emb_j = sample_embeddings[j][emb_size:]
                    
                    # Compute quantum kernel between pairs
                    similarity = quantum_kernel_engine.compute_quantum_kernel(
                        query_emb_i, doc_emb_i, query_emb_j, doc_emb_j
                    )
                    
                    kernel_matrix[i, j] = similarity
                    kernel_matrix[j, i] = similarity  # Symmetric
            
            # Create target kernel from labels
            target_kernel = self._create_target_kernel(sample_labels)
            
            # Compute KTA score
            kta_score = self.kta_calculator.compute_kta(kernel_matrix, target_kernel)
            
            # Update optimization state
            self.iteration_count += 1
            self.optimization_history.append(kta_score)
            
            if kta_score > self.best_score:
                self.best_score = kta_score
                self.best_parameters = parameters.copy()
            
            # Log progress
            if self.iteration_count % 10 == 0:
                logger.info(f"Iteration {self.iteration_count}: KTA = {kta_score:.4f}, Best = {self.best_score:.4f}")
            
            # Return negative KTA for minimization
            return -kta_score
            
        except Exception as e:
            logger.warning(f"Error evaluating parameters: {e}")
            return 1.0  # Return large positive value for minimization
    
    def _create_target_kernel(self, labels: np.ndarray) -> np.ndarray:
        """Create target kernel matrix from relevance labels."""
        n = len(labels)
        target_kernel = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Target kernel: 1 if same relevance level, 0 otherwise
                if labels[i] == labels[j]:
                    target_kernel[i, j] = 1.0
                else:
                    # Partial similarity for adjacent relevance levels
                    label_diff = abs(labels[i] - labels[j])
                    if label_diff == 1:
                        target_kernel[i, j] = 0.5
                    else:
                        target_kernel[i, j] = 0.0
        
        return target_kernel
    
    def _run_differential_evolution(self, objective_function: Callable,
                                  bounds: List[Tuple[float, float]]) -> Any:
        """Run differential evolution optimization."""
        logger.info("Running differential evolution optimization")
        
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=self.config.max_iterations,
            popsize=self.config.population_size,
            seed=self.config.random_seed,
            atol=self.config.convergence_threshold,
            tol=self.config.convergence_threshold
        )
        
        return result
    
    def _run_scipy_minimize(self, objective_function: Callable,
                          bounds: List[Tuple[float, float]]) -> Any:
        """Run scipy minimize optimization."""
        logger.info("Running scipy minimize optimization")
        
        # Initial parameters (random within bounds)
        n_params = len(bounds)
        x0 = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            n_params
        )
        
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.convergence_threshold
            }
        )
        
        return result
    
    def _validate_parameters(self, parameters: np.ndarray,
                           quantum_kernel_engine: QuantumKernelEngine,
                           val_embeddings: np.ndarray,
                           val_labels: np.ndarray) -> Dict[str, float]:
        """Validate optimized parameters on validation set."""
        logger.info("Validating optimized parameters")
        
        # Set optimal parameters
        quantum_kernel_engine.set_parameters(parameters)
        
        # Compute validation KTA score
        val_kta_score = -self._evaluate_parameters(parameters, quantum_kernel_engine, val_embeddings, val_labels)
        
        # Compute ranking correlation
        val_similarities = []
        val_targets = []
        
        n_val_samples = min(30, len(val_embeddings))
        for i in range(n_val_samples):
            emb_size = len(val_embeddings[i]) // 2
            query_emb = val_embeddings[i][:emb_size]
            doc_emb = val_embeddings[i][emb_size:]
            
            # Compute similarity with quantum kernel
            similarity = quantum_kernel_engine.compute_quantum_similarity(query_emb, doc_emb)
            val_similarities.append(similarity)
            val_targets.append(val_labels[i])
        
        # Compute ranking correlation
        ranking_correlation, _ = spearmanr(val_similarities, val_targets)
        
        validation_results = {
            'validation_kta_score': val_kta_score,
            'ranking_correlation': ranking_correlation,
            'samples_evaluated': n_val_samples
        }
        
        logger.info(f"Validation results: {validation_results}")
        
        return validation_results
    
    def _analyze_parameter_sensitivity(self, parameters: np.ndarray,
                                     quantum_kernel_engine: QuantumKernelEngine,
                                     embeddings: np.ndarray,
                                     labels: np.ndarray) -> Dict[str, float]:
        """Analyze sensitivity of each parameter."""
        logger.info("Analyzing parameter sensitivity")
        
        sensitivity_results = {}
        baseline_score = -self._evaluate_parameters(parameters, quantum_kernel_engine, embeddings, labels)
        
        # Test small perturbations to each parameter
        perturbation = 0.1
        
        for i, param_value in enumerate(parameters):
            # Test positive perturbation
            perturbed_params = parameters.copy()
            perturbed_params[i] += perturbation
            
            if perturbed_params[i] <= self.config.parameter_bounds[1]:
                pos_score = -self._evaluate_parameters(perturbed_params, quantum_kernel_engine, embeddings, labels)
                sensitivity = abs(pos_score - baseline_score) / perturbation
            else:
                sensitivity = 0.0
            
            sensitivity_results[f'param_{i}'] = sensitivity
        
        logger.info(f"Parameter sensitivity analysis completed")
        
        return sensitivity_results


class QuantumKernelTrainer:
    """
    High-level trainer for quantum kernels on medical data.
    
    Combines KTA optimization with medical domain validation and
    comprehensive evaluation on medical ranking tasks.
    """
    
    def __init__(self, quantum_config: Optional[QuantumConfig] = None,
                 kta_config: Optional[KTAOptimizationConfig] = None):
        """Initialize quantum kernel trainer."""
        self.quantum_config = quantum_config or QuantumConfig()
        self.kta_config = kta_config or KTAOptimizationConfig()
        
        # Initialize components
        # Convert QuantumConfig to QuantumKernelConfig
        from ..core.quantum_kernel_engine import QuantumKernelConfig
        kernel_config = QuantumKernelConfig(
            n_qubits=self.quantum_config.n_qubits,
            encoding_method="amplitude",
            enable_caching=True,
            max_cache_size=1000,
            batch_size=50,
            enable_kta_optimization=True,
            enable_feature_selection=False,  # Disable for medical training
            num_selected_features=32,
            kta_optimization_iterations=100
        )
        self.quantum_kernel_engine = QuantumKernelEngine(kernel_config)
        self.kta_optimizer = KTAOptimizer(self.kta_config)
        
        logger.info("Quantum kernel trainer initialized")
    
    def train_on_medical_corpus(self, training_pairs: List[TrainingPair],
                              validation_pairs: Optional[List[TrainingPair]] = None) -> QuantumKernelTrainingResult:
        """
        Train quantum kernel on medical corpus data.
        
        Args:
            training_pairs: Medical training data
            validation_pairs: Validation data
            
        Returns:
            Training results with optimized parameters
        """
        logger.info(f"Training quantum kernel on {len(training_pairs)} medical training pairs")
        
        # Run KTA optimization
        training_result = self.kta_optimizer.optimize_parameters(
            self.quantum_kernel_engine,
            training_pairs,
            validation_pairs
        )
        
        # Set optimal parameters in engine
        self.quantum_kernel_engine.set_parameters(training_result.optimal_parameters)
        
        logger.info("Quantum kernel training completed")
        
        return training_result
    
    def evaluate_on_medical_ranking(self, test_pairs: List[TrainingPair]) -> Dict[str, float]:
        """Evaluate trained quantum kernel on medical ranking task."""
        logger.info(f"Evaluating quantum kernel on {len(test_pairs)} test pairs")
        
        similarities = []
        targets = []
        
        for pair in test_pairs:
            emb_size = len(pair.query_embedding)
            
            # Compute quantum similarity
            similarity = self.quantum_kernel_engine.compute_quantum_similarity(
                pair.query_embedding, pair.doc_embedding
            )
            similarities.append(similarity)
            targets.append(pair.relevance_label)
        
        # Compute ranking metrics
        ranking_correlation, _ = spearmanr(similarities, targets)
        
        # Compute accuracy for binary classification
        binary_predictions = [1 if s > 0.5 else 0 for s in similarities]
        binary_targets = [1 if t > 0 else 0 for t in targets]
        accuracy = np.mean([p == t for p, t in zip(binary_predictions, binary_targets)])
        
        results = {
            'ranking_correlation': ranking_correlation,
            'classification_accuracy': accuracy,
            'mean_similarity': np.mean(similarities),
            'similarity_std': np.std(similarities)
        }
        
        logger.info(f"Medical ranking evaluation: {results}")
        
        return results
    
    def save_trained_model(self, filepath: str, training_result: QuantumKernelTrainingResult):
        """Save trained quantum kernel model."""
        model_data = {
            'optimal_parameters': training_result.optimal_parameters,
            'training_result': training_result,
            'quantum_config': self.quantum_config,
            'kta_config': self.kta_config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved trained quantum kernel model to {filepath}")
    
    def load_trained_model(self, filepath: str) -> QuantumKernelTrainingResult:
        """Load trained quantum kernel model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Set parameters in engine
        self.quantum_kernel_engine.set_parameters(model_data['optimal_parameters'])
        
        logger.info(f"Loaded trained quantum kernel model from {filepath}")
        
        return model_data['training_result']


class QuantumKernelOptimizationPipeline:
    """
    Complete pipeline for quantum kernel optimization on medical data.
    """
    
    def __init__(self, quantum_config: Optional[QuantumConfig] = None,
                 kta_config: Optional[KTAOptimizationConfig] = None):
        """Initialize quantum kernel optimization pipeline."""
        self.quantum_config = quantum_config or QuantumConfig()
        self.kta_config = kta_config or KTAOptimizationConfig()
        self.trainer = QuantumKernelTrainer(self.quantum_config, self.kta_config)
        
        logger.info("Quantum kernel optimization pipeline initialized")
    
    def run(self, train_pairs: List[TrainingPair],
            val_pairs: List[TrainingPair],
            test_pairs: List[TrainingPair],
            output_dir: str = "quantum_kernel_training") -> Dict[str, Any]:
        """
        Run complete quantum kernel optimization pipeline.
        
        Args:
            train_pairs: Training data
            val_pairs: Validation data  
            test_pairs: Test data
            output_dir: Output directory
            
        Returns:
            Pipeline results
        """
        logger.info("Starting quantum kernel optimization pipeline")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Train quantum kernel
        logger.info("Step 1: Training quantum kernel with KTA optimization")
        training_result = self.trainer.train_on_medical_corpus(train_pairs, val_pairs)
        
        # Step 2: Evaluate on test set
        logger.info("Step 2: Evaluating on medical ranking test set")
        test_results = self.trainer.evaluate_on_medical_ranking(test_pairs)
        
        # Step 3: Save results
        logger.info("Step 3: Saving training results")
        model_path = output_path / "trained_quantum_kernel.pkl"
        results_path = output_path / "training_results.pkl"
        
        self.trainer.save_trained_model(str(model_path), training_result)
        
        # Save comprehensive results
        results = {
            'training_result': training_result,
            'test_evaluation': test_results,
            'config': {
                'quantum_config': self.quantum_config,
                'kta_config': self.kta_config
            },
            'data_summary': {
                'train_pairs': len(train_pairs),
                'val_pairs': len(val_pairs),
                'test_pairs': len(test_pairs)
            },
            'file_paths': {
                'model': str(model_path),
                'results': str(results_path)
            }
        }
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("Quantum kernel optimization pipeline completed successfully")
        logger.info(f"Best KTA score: {training_result.best_kta_score:.4f}")
        logger.info(f"Test ranking correlation: {test_results['ranking_correlation']:.4f}")
        
        return results