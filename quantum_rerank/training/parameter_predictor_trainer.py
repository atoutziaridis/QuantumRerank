"""
Medical Domain Parameter Predictor Training.

This module implements training pipelines for the quantum parameter predictor
on medical embeddings, optimizing the MLP to predict quantum circuit parameters
that maximize performance on medical document ranking tasks.

Based on QRF-04 requirements for parameter predictor training pipeline.
"""

import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from .medical_data_preparation import TrainingPair
from .quantum_kernel_trainer import QuantumKernelTrainingResult
from ..ml.parameter_predictor import QuantumParameterPredictor, ParameterPredictorConfig
from ..core.quantum_kernel_engine import QuantumKernelEngine
from ..config.settings import QuantumConfig

logger = logging.getLogger(__name__)


@dataclass
class ParameterPredictorTrainingConfig:
    """Configuration for parameter predictor training."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    validation_frequency: int = 5
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, none
    loss_function: str = "mse"  # mse, huber, cosine_similarity
    gradient_clip: float = 1.0
    save_best_model: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42


@dataclass
class ParameterPredictorTrainingResult:
    """Results from parameter predictor training."""
    best_model_state: Dict[str, torch.Tensor]
    training_loss_history: List[float]
    validation_loss_history: List[float]
    correlation_metrics: Dict[str, float]
    convergence_epoch: int
    training_time_seconds: float
    architecture_metrics: Dict[str, Any]
    medical_domain_performance: Dict[str, float]


class MedicalParameterPredictorTrainer:
    """
    Trainer for quantum parameter predictor on medical embeddings.
    
    Trains the MLP to predict optimal quantum circuit parameters from
    medical document embeddings, optimizing for medical ranking performance.
    """
    
    def __init__(self, predictor_config: Optional[ParameterPredictorConfig] = None,
                 training_config: Optional[ParameterPredictorTrainingConfig] = None):
        """Initialize parameter predictor trainer."""
        self.predictor_config = predictor_config or ParameterPredictorConfig()
        self.training_config = training_config or ParameterPredictorTrainingConfig()
        
        # Initialize predictor
        self.predictor = QuantumParameterPredictor(self.predictor_config)
        self.predictor.to(self.training_config.device)
        
        # Training state
        self.best_model_state = None
        self.best_validation_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Set random seeds
        torch.manual_seed(self.training_config.random_seed)
        np.random.seed(self.training_config.random_seed)
        
        logger.info(f"Medical parameter predictor trainer initialized on {self.training_config.device}")
    
    def train_on_medical_embeddings(self, training_pairs: List[TrainingPair],
                                  optimal_parameters: np.ndarray,
                                  validation_pairs: Optional[List[TrainingPair]] = None,
                                  validation_parameters: Optional[np.ndarray] = None) -> ParameterPredictorTrainingResult:
        """
        Train parameter predictor on medical embeddings.
        
        Args:
            training_pairs: Training data with medical embeddings
            optimal_parameters: Optimal quantum parameters from KTA optimization
            validation_pairs: Validation data
            validation_parameters: Validation optimal parameters
            
        Returns:
            Training results with best model and metrics
        """
        logger.info(f"Training parameter predictor on {len(training_pairs)} medical pairs")
        start_time = time.time()
        
        # Prepare data loaders
        train_loader = self._prepare_data_loader(training_pairs, optimal_parameters)
        val_loader = None
        if validation_pairs and validation_parameters is not None:
            val_loader = self._prepare_data_loader(validation_pairs, validation_parameters)
        
        # Initialize optimizer and scheduler
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        criterion = self._create_loss_function()
        
        # Training history
        training_loss_history = []
        validation_loss_history = []
        
        # Training loop
        for epoch in range(self.training_config.num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            training_loss_history.append(train_loss)
            
            # Validation phase
            if val_loader and epoch % self.training_config.validation_frequency == 0:
                val_loss, val_metrics = self._validate_epoch(val_loader, criterion)
                validation_loss_history.append(val_loss)
                
                # Update learning rate scheduler
                if scheduler and self.training_config.scheduler == "reduce_on_plateau":
                    scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < self.best_validation_loss:
                    self.best_validation_loss = val_loss
                    self.best_model_state = self.predictor.state_dict()
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                logger.info(f"Epoch {epoch+1}/{self.training_config.num_epochs}: "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Val Correlation: {val_metrics['pearson_r']:.4f}")
                
                if self.epochs_without_improvement >= self.training_config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{self.training_config.num_epochs}: "
                          f"Train Loss: {train_loss:.4f}")
            
            # Update cosine scheduler
            if scheduler and self.training_config.scheduler == "cosine":
                scheduler.step()
        
        # Load best model
        if self.best_model_state:
            self.predictor.load_state_dict(self.best_model_state)
        
        # Final evaluation
        correlation_metrics = self._evaluate_correlation(validation_pairs, validation_parameters)
        architecture_metrics = self._analyze_architecture_performance()
        medical_domain_performance = self._evaluate_medical_domains(validation_pairs, validation_parameters)
        
        training_time = time.time() - start_time
        
        result = ParameterPredictorTrainingResult(
            best_model_state=self.best_model_state or self.predictor.state_dict(),
            training_loss_history=training_loss_history,
            validation_loss_history=validation_loss_history,
            correlation_metrics=correlation_metrics,
            convergence_epoch=epoch + 1,
            training_time_seconds=training_time,
            architecture_metrics=architecture_metrics,
            medical_domain_performance=medical_domain_performance
        )
        
        logger.info(f"Parameter predictor training completed in {training_time:.2f}s")
        logger.info(f"Best validation loss: {self.best_validation_loss:.4f}")
        logger.info(f"Correlation metrics: {correlation_metrics}")
        
        return result
    
    def _prepare_data_loader(self, training_pairs: List[TrainingPair],
                           optimal_parameters: np.ndarray) -> DataLoader:
        """Prepare PyTorch data loader from training pairs."""
        # Extract embeddings
        embeddings = []
        for pair in training_pairs:
            # Use query embedding as input (could also concatenate query+doc)
            embeddings.append(pair.query_embedding)
        
        embeddings_tensor = torch.FloatTensor(np.array(embeddings))
        parameters_tensor = torch.FloatTensor(optimal_parameters)
        
        # Ensure parameters match the number of training pairs
        if len(parameters_tensor.shape) == 1:
            # Replicate parameters for each training pair if single set
            parameters_tensor = parameters_tensor.unsqueeze(0).repeat(len(training_pairs), 1)
        
        dataset = TensorDataset(embeddings_tensor, parameters_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        return loader
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.training_config.optimizer == "adam":
            return optim.Adam(
                self.predictor.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == "adamw":
            return optim.AdamW(
                self.predictor.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == "sgd":
            return optim.SGD(
                self.predictor.parameters(),
                lr=self.training_config.learning_rate,
                momentum=0.9,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler."""
        if self.training_config.scheduler == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        elif self.training_config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.training_config.num_epochs
            )
        elif self.training_config.scheduler == "none":
            return None
        else:
            return None
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        if self.training_config.loss_function == "mse":
            return nn.MSELoss()
        elif self.training_config.loss_function == "huber":
            return nn.HuberLoss()
        elif self.training_config.loss_function == "cosine_similarity":
            return nn.CosineEmbeddingLoss()
        else:
            return nn.MSELoss()
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.predictor.train()
        total_loss = 0.0
        
        for embeddings, target_params in train_loader:
            embeddings = embeddings.to(self.training_config.device)
            target_params = target_params.to(self.training_config.device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_params = self.predictor.get_flat_parameters(embeddings)
            
            # Ensure shapes match
            if predicted_params.shape != target_params.shape:
                # Truncate or pad to match
                min_params = min(predicted_params.shape[1], target_params.shape[1])
                predicted_params = predicted_params[:, :min_params]
                target_params = target_params[:, :min_params]
            
            loss = criterion(predicted_params, target_params)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.predictor.parameters(),
                    self.training_config.gradient_clip
                )
            
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader,
                       criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.predictor.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for embeddings, target_params in val_loader:
                embeddings = embeddings.to(self.training_config.device)
                target_params = target_params.to(self.training_config.device)
                
                predicted_params = self.predictor.get_flat_parameters(embeddings)
                
                # Ensure shapes match
                if predicted_params.shape != target_params.shape:
                    min_params = min(predicted_params.shape[1], target_params.shape[1])
                    predicted_params = predicted_params[:, :min_params]
                    target_params = target_params[:, :min_params]
                
                loss = criterion(predicted_params, target_params)
                total_loss += loss.item()
                
                all_predictions.append(predicted_params.cpu().numpy())
                all_targets.append(target_params.cpu().numpy())
        
        # Calculate correlation metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        pearson_r, _ = pearsonr(all_predictions.flatten(), all_targets.flatten())
        spearman_r, _ = spearmanr(all_predictions.flatten(), all_targets.flatten())
        
        metrics = {
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            'mse': mean_squared_error(all_targets, all_predictions),
            'r2': r2_score(all_targets, all_predictions)
        }
        
        return total_loss / len(val_loader), metrics
    
    def _evaluate_correlation(self, validation_pairs: Optional[List[TrainingPair]],
                            validation_parameters: Optional[np.ndarray]) -> Dict[str, float]:
        """Evaluate correlation between predicted and optimal parameters."""
        if not validation_pairs or validation_parameters is None:
            return {}
        
        self.predictor.eval()
        
        # Get embeddings
        embeddings = torch.FloatTensor([pair.query_embedding for pair in validation_pairs])
        embeddings = embeddings.to(self.training_config.device)
        
        with torch.no_grad():
            predicted_params = self.predictor.get_flat_parameters(embeddings)
            predicted_params = predicted_params.cpu().numpy()
        
        # Ensure shapes match
        if predicted_params.shape != validation_parameters.shape:
            min_params = min(predicted_params.shape[1], validation_parameters.shape[1])
            predicted_params = predicted_params[:, :min_params]
            validation_parameters = validation_parameters[:, :min_params]
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(predicted_params.flatten(), validation_parameters.flatten())
        spearman_r, spearman_p = spearmanr(predicted_params.flatten(), validation_parameters.flatten())
        
        # Parameter-wise correlations
        param_correlations = []
        for i in range(predicted_params.shape[1]):
            r, _ = pearsonr(predicted_params[:, i], validation_parameters[:, i])
            param_correlations.append(r)
        
        return {
            'overall_pearson_r': pearson_r,
            'overall_pearson_p': pearson_p,
            'overall_spearman_r': spearman_r,
            'overall_spearman_p': spearman_p,
            'mean_param_correlation': np.mean(param_correlations),
            'std_param_correlation': np.std(param_correlations),
            'min_param_correlation': np.min(param_correlations),
            'max_param_correlation': np.max(param_correlations)
        }
    
    def _analyze_architecture_performance(self) -> Dict[str, Any]:
        """Analyze predictor architecture performance."""
        # Count parameters
        total_params = sum(p.numel() for p in self.predictor.parameters())
        trainable_params = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        
        # Analyze layer-wise parameter distribution
        layer_params = {}
        for name, param in self.predictor.named_parameters():
            layer_params[name] = {
                'shape': list(param.shape),
                'params': param.numel(),
                'mean': float(param.data.mean()),
                'std': float(param.data.std())
            }
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'embedding_dim': self.predictor_config.embedding_dim,
                'hidden_dims': self.predictor_config.hidden_dims,
                'n_qubits': self.predictor_config.n_qubits,
                'n_layers': self.predictor_config.n_layers,
                'dropout_rate': self.predictor_config.dropout_rate
            },
            'layer_analysis': layer_params
        }
    
    def _evaluate_medical_domains(self, validation_pairs: Optional[List[TrainingPair]],
                                validation_parameters: Optional[np.ndarray]) -> Dict[str, float]:
        """Evaluate performance across different medical domains."""
        if not validation_pairs or validation_parameters is None:
            return {}
        
        # Group by medical domain
        domain_pairs = {}
        domain_params = {}
        
        for i, pair in enumerate(validation_pairs):
            domain = pair.medical_domain
            if domain not in domain_pairs:
                domain_pairs[domain] = []
                domain_params[domain] = []
            
            domain_pairs[domain].append(pair)
            if len(validation_parameters.shape) == 1:
                domain_params[domain].append(validation_parameters)
            else:
                domain_params[domain].append(validation_parameters[i])
        
        # Evaluate per domain
        domain_performance = {}
        
        for domain, pairs in domain_pairs.items():
            embeddings = torch.FloatTensor([pair.query_embedding for pair in pairs])
            embeddings = embeddings.to(self.training_config.device)
            
            with torch.no_grad():
                predicted = self.predictor.get_flat_parameters(embeddings).cpu().numpy()
            
            target = np.array(domain_params[domain])
            if len(target.shape) == 1:
                target = target.reshape(1, -1).repeat(len(pairs), axis=0)
            
            # Ensure shapes match
            if predicted.shape != target.shape:
                min_params = min(predicted.shape[1], target.shape[1])
                predicted = predicted[:, :min_params]
                target = target[:, :min_params]
            
            r, _ = pearsonr(predicted.flatten(), target.flatten())
            mse = mean_squared_error(target, predicted)
            
            domain_performance[domain] = {
                'correlation': r,
                'mse': mse,
                'n_samples': len(pairs)
            }
        
        return domain_performance
    
    def optimize_architecture(self, training_pairs: List[TrainingPair],
                            optimal_parameters: np.ndarray,
                            architectures_to_test: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Optimize MLP architecture for medical domain.
        
        Args:
            training_pairs: Training data
            optimal_parameters: Target parameters
            architectures_to_test: List of architectures to test
            
        Returns:
            Best architecture and performance metrics
        """
        logger.info("Optimizing parameter predictor architecture")
        
        if architectures_to_test is None:
            # Default architectures to test
            architectures_to_test = [
                {'hidden_dims': [256, 128]},
                {'hidden_dims': [512, 256]},
                {'hidden_dims': [512, 256, 128]},
                {'hidden_dims': [768, 384, 192]},
                {'hidden_dims': [1024, 512, 256]}
            ]
        
        best_architecture = None
        best_performance = float('-inf')
        architecture_results = []
        
        # Split data for architecture search
        n_train = int(0.8 * len(training_pairs))
        arch_train_pairs = training_pairs[:n_train]
        arch_val_pairs = training_pairs[n_train:]
        
        for arch_config in architectures_to_test:
            logger.info(f"Testing architecture: {arch_config}")
            
            # Create new predictor with architecture
            test_config = ParameterPredictorConfig(
                embedding_dim=self.predictor_config.embedding_dim,
                hidden_dims=arch_config['hidden_dims'],
                n_qubits=self.predictor_config.n_qubits,
                n_layers=self.predictor_config.n_layers,
                dropout_rate=arch_config.get('dropout_rate', 0.1)
            )
            
            # Train with limited epochs for architecture search
            quick_train_config = ParameterPredictorTrainingConfig(
                num_epochs=30,
                early_stopping_patience=5,
                device=self.training_config.device
            )
            
            # Create and train predictor
            test_trainer = MedicalParameterPredictorTrainer(test_config, quick_train_config)
            result = test_trainer.train_on_medical_embeddings(
                arch_train_pairs, optimal_parameters,
                arch_val_pairs, optimal_parameters
            )
            
            # Evaluate performance
            performance = result.correlation_metrics.get('overall_pearson_r', 0)
            
            architecture_results.append({
                'architecture': arch_config,
                'performance': performance,
                'training_time': result.training_time_seconds,
                'convergence_epoch': result.convergence_epoch
            })
            
            if performance > best_performance:
                best_performance = performance
                best_architecture = arch_config
        
        logger.info(f"Best architecture: {best_architecture} with correlation {best_performance:.4f}")
        
        return {
            'best_architecture': best_architecture,
            'best_performance': best_performance,
            'all_results': architecture_results
        }
    
    def save_trained_model(self, filepath: str, training_result: ParameterPredictorTrainingResult):
        """Save trained parameter predictor model."""
        model_data = {
            'model_state_dict': training_result.best_model_state,
            'predictor_config': self.predictor_config,
            'training_config': self.training_config,
            'training_result': training_result
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Saved trained parameter predictor to {filepath}")
    
    def load_trained_model(self, filepath: str) -> ParameterPredictorTrainingResult:
        """Load trained parameter predictor model."""
        model_data = torch.load(filepath, map_location=self.training_config.device)
        
        # Load model state
        self.predictor.load_state_dict(model_data['model_state_dict'])
        self.predictor.to(self.training_config.device)
        
        logger.info(f"Loaded trained parameter predictor from {filepath}")
        
        return model_data['training_result']


class ParameterPredictorTrainingPipeline:
    """
    Complete pipeline for training parameter predictor on medical embeddings.
    """
    
    def __init__(self, predictor_config: Optional[ParameterPredictorConfig] = None,
                 training_config: Optional[ParameterPredictorTrainingConfig] = None):
        """Initialize parameter predictor training pipeline."""
        self.predictor_config = predictor_config or ParameterPredictorConfig()
        self.training_config = training_config or ParameterPredictorTrainingConfig()
        
        logger.info("Parameter predictor training pipeline initialized")
    
    def run(self, training_pairs: List[TrainingPair],
            optimal_parameters: np.ndarray,
            validation_pairs: List[TrainingPair],
            test_pairs: List[TrainingPair],
            output_dir: str = "parameter_predictor_training") -> Dict[str, Any]:
        """
        Run complete parameter predictor training pipeline.
        
        Args:
            training_pairs: Training data
            optimal_parameters: Optimal parameters from KTA
            validation_pairs: Validation data
            test_pairs: Test data
            output_dir: Output directory
            
        Returns:
            Pipeline results
        """
        logger.info("Starting parameter predictor training pipeline")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Architecture optimization
        logger.info("Step 1: Optimizing architecture for medical domain")
        trainer = MedicalParameterPredictorTrainer(self.predictor_config, self.training_config)
        
        architecture_results = trainer.optimize_architecture(
            training_pairs[:1000],  # Use subset for architecture search
            optimal_parameters
        )
        
        # Update config with best architecture
        if architecture_results['best_architecture']:
            self.predictor_config.hidden_dims = architecture_results['best_architecture']['hidden_dims']
        
        # Step 2: Train with best architecture
        logger.info("Step 2: Training with optimized architecture")
        trainer = MedicalParameterPredictorTrainer(self.predictor_config, self.training_config)
        
        training_result = trainer.train_on_medical_embeddings(
            training_pairs, optimal_parameters,
            validation_pairs, optimal_parameters
        )
        
        # Step 3: Evaluate on test set
        logger.info("Step 3: Evaluating on test set")
        test_metrics = trainer._evaluate_correlation(test_pairs, optimal_parameters)
        
        # Step 4: Save results
        logger.info("Step 4: Saving training results")
        model_path = output_path / "trained_parameter_predictor.pth"
        results_path = output_path / "training_results.pkl"
        
        trainer.save_trained_model(str(model_path), training_result)
        
        # Save comprehensive results
        results = {
            'training_result': training_result,
            'architecture_optimization': architecture_results,
            'test_metrics': test_metrics,
            'config': {
                'predictor_config': self.predictor_config,
                'training_config': self.training_config
            },
            'data_summary': {
                'train_pairs': len(training_pairs),
                'val_pairs': len(validation_pairs),
                'test_pairs': len(test_pairs)
            },
            'file_paths': {
                'model': str(model_path),
                'results': str(results_path)
            }
        }
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("Parameter predictor training pipeline completed successfully")
        logger.info(f"Best correlation: {training_result.correlation_metrics.get('overall_pearson_r', 0):.4f}")
        logger.info(f"Test correlation: {test_metrics.get('overall_pearson_r', 0):.4f}")
        
        return results