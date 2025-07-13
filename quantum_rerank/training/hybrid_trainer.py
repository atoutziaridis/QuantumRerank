"""
Hybrid Quantum-Classical Trainer for QuantumRerank.

This module implements the main hybrid training architecture combining quantum
parameterized circuits with classical neural networks for optimized similarity
computation and ranking performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
from datetime import datetime

from ..quantum import QuantumSimilarityEngine
from ..ml import QuantumParameterPredictor
from ..config import Configurable, QuantumRerankConfigSchema
from ..utils import get_logger, QuantumRerankException
from .loss_functions import QuantumAwareLoss, AdaptiveLossScheduler
from .optimizers import HybridOptimizer


@dataclass
class HybridTrainingConfig:
    """Configuration for hybrid quantum-classical training."""
    # Quantum parameters
    n_qubits: int = 4
    circuit_depth: int = 15
    parameter_count: int = 24
    quantum_learning_rate: float = 0.01
    
    # Classical parameters
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.1
    classical_learning_rate: float = 0.001
    
    # Training parameters
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Optimization parameters
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0
    
    # Loss function parameters
    triplet_margin: float = 0.5
    fidelity_weight: float = 0.1
    parameter_reg_weight: float = 0.01
    
    # Performance targets
    target_ranking_improvement: float = 0.15  # PRD: 10-20% improvement
    max_training_time_hours: float = 24.0     # Practical training time
    memory_limit_gb: float = 8.0              # Training resource limit


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring and evaluation."""
    epoch: int = 0
    training_loss: float = 0.0
    validation_loss: float = 0.0
    ranking_accuracy: float = 0.0
    margin_satisfaction: float = 0.0
    avg_quantum_fidelity: float = 0.0
    training_time: float = 0.0
    memory_usage_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "epoch": self.epoch,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "ranking_accuracy": self.ranking_accuracy,
            "margin_satisfaction": self.margin_satisfaction,
            "avg_quantum_fidelity": self.avg_quantum_fidelity,
            "training_time": self.training_time,
            "memory_usage_gb": self.memory_usage_gb
        }


class QuantumCircuitTrainer(nn.Module):
    """Quantum parameter optimization for similarity circuits."""
    
    def __init__(self, n_qubits: int, circuit_depth: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.logger = get_logger(__name__)
        
        # Initialize quantum parameters
        self.quantum_params = nn.Parameter(
            torch.randn(circuit_depth * n_qubits) * 0.1
        )
        
        # Quantum similarity engine
        self.quantum_engine = QuantumSimilarityEngine()
    
    def forward(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through quantum similarity circuit.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            
        Returns:
            Tuple of (similarity_score, quantum_fidelity)
        """
        batch_size = embedding1.size(0)
        similarities = []
        fidelities = []
        
        for i in range(batch_size):
            emb1 = embedding1[i].detach().cpu().numpy()
            emb2 = embedding2[i].detach().cpu().numpy()
            
            # Compute quantum similarity with current parameters
            similarity = self.quantum_engine.compute_similarity(
                emb1, emb2, 
                quantum_params=self.quantum_params.detach().cpu().numpy()
            )
            
            # Compute quantum fidelity for regularization
            fidelity = self._compute_quantum_fidelity(emb1, emb2)
            
            similarities.append(similarity)
            fidelities.append(fidelity)
        
        similarities = torch.tensor(similarities, dtype=torch.float32, device=embedding1.device)
        fidelities = torch.tensor(fidelities, dtype=torch.float32, device=embedding1.device)
        
        return similarities, fidelities
    
    def _compute_quantum_fidelity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute quantum state fidelity."""
        try:
            # Normalize embeddings
            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # Compute fidelity as overlap squared
            fidelity = np.abs(np.dot(emb1, emb2)) ** 2
            return float(fidelity)
        
        except Exception as e:
            self.logger.warning(f"Failed to compute quantum fidelity: {e}")
            return 0.5  # Default moderate fidelity
    
    def get_parameter_gradients(self) -> Optional[torch.Tensor]:
        """Get gradients for quantum parameters."""
        if self.quantum_params.grad is not None:
            return self.quantum_params.grad.clone()
        return None


class ClassicalMLPTrainer(nn.Module):
    """Classical neural network for parameter prediction."""
    
    def __init__(self, embedding_dim: int, quantum_param_count: int, hidden_dims: List[int]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.quantum_param_count = quantum_param_count
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim * 2  # Concatenated embeddings
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer for quantum parameters
        layers.append(nn.Linear(input_dim, quantum_param_count))
        layers.append(nn.Tanh())  # Bound parameters to [-1, 1]
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """
        Predict quantum circuit parameters from embeddings.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            
        Returns:
            Predicted quantum parameters
        """
        # Concatenate embeddings
        combined_embeddings = torch.cat([embedding1, embedding2], dim=-1)
        
        # Forward pass through MLP
        quantum_params = self.mlp(combined_embeddings)
        
        # Scale parameters appropriately for quantum circuits
        quantum_params = quantum_params * np.pi  # Scale to [0, 2π]
        
        return quantum_params


class HybridQuantumClassicalTrainer(Configurable):
    """
    Main hybrid training class combining quantum and classical components.
    
    This trainer implements the complete hybrid architecture for quantum-enhanced
    similarity learning with classical neural network integration.
    """
    
    def __init__(self, config: HybridTrainingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Training state
        self.current_epoch = 0
        self.best_validation_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history: List[TrainingMetrics] = []
        self.start_time = None
        
        # Initialize quantum trainer
        self.quantum_trainer = QuantumCircuitTrainer(
            n_qubits=config.n_qubits,
            circuit_depth=config.circuit_depth
        )
        
        # Initialize classical trainer
        self.classical_trainer = ClassicalMLPTrainer(
            embedding_dim=768,  # Default embedding dimension
            quantum_param_count=config.parameter_count,
            hidden_dims=config.hidden_layers
        )
        
        # Initialize loss function
        self.loss_function = QuantumAwareLoss(
            triplet_weight=1.0,
            fidelity_weight=config.fidelity_weight,
            parameter_reg_weight=config.parameter_reg_weight,
            margin=config.triplet_margin
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss scheduler
        self.loss_scheduler = AdaptiveLossScheduler(
            initial_weights={
                "triplet_loss": 1.0,
                "fidelity_loss": config.fidelity_weight,
                "parameter_reg_loss": config.parameter_reg_weight
            }
        )
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantum_trainer.to(self.device)
        self.classical_trainer.to(self.device)
        
        self.logger.info(f"Initialized hybrid trainer on device: {self.device}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for hybrid model parameters."""
        # Combine parameters from both models
        all_params = list(self.quantum_trainer.parameters()) + list(self.classical_trainer.parameters())
        
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                all_params,
                lr=self.config.classical_learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                all_params,
                lr=self.config.classical_learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def hybrid_training_step(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Execute single hybrid training step.
        
        Args:
            anchor_embeddings: Anchor embedding batch
            positive_embeddings: Positive embedding batch  
            negative_embeddings: Negative embedding batch
            
        Returns:
            Tuple of (total_loss, loss_components, metrics)
        """
        # 1. Classical parameter prediction
        quantum_params_pos = self.classical_trainer(anchor_embeddings, positive_embeddings)
        quantum_params_neg = self.classical_trainer(anchor_embeddings, negative_embeddings)
        
        # 2. Quantum similarity computation
        pos_similarity, pos_fidelity = self.quantum_trainer(anchor_embeddings, positive_embeddings)
        neg_similarity, neg_fidelity = self.quantum_trainer(anchor_embeddings, negative_embeddings)
        anchor_similarity, anchor_fidelity = self.quantum_trainer(anchor_embeddings, anchor_embeddings)
        
        # 3. Compute loss
        total_loss, loss_components = self.loss_function(
            anchor_similarity=anchor_similarity,
            positive_similarity=pos_similarity,
            negative_similarity=neg_similarity,
            quantum_parameters=torch.cat([quantum_params_pos, quantum_params_neg], dim=0),
            quantum_fidelities=torch.cat([pos_fidelity, neg_fidelity], dim=0)
        )
        
        # 4. Compute training metrics
        metrics = self._compute_training_metrics(
            anchor_similarity, pos_similarity, neg_similarity,
            torch.cat([pos_fidelity, neg_fidelity], dim=0)
        )
        
        return total_loss, loss_components, metrics
    
    def train_epoch(
        self,
        train_dataloader: torch.utils.data.DataLoader
    ) -> TrainingMetrics:
        """Train for one epoch."""
        self.quantum_trainer.train()
        self.classical_trainer.train()
        
        epoch_start_time = time.time()
        total_loss = 0.0
        total_metrics = {
            "ranking_accuracy": 0.0,
            "margin_satisfaction": 0.0,
            "avg_quantum_fidelity": 0.0
        }
        num_batches = 0
        
        for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
            # Move to device
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss, loss_components, batch_metrics = self.hybrid_training_step(
                anchor, positive, negative
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.quantum_trainer.parameters()) + 
                    list(self.classical_trainer.parameters()),
                    self.config.gradient_clipping
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            for key in total_metrics:
                if key in batch_metrics:
                    total_metrics[key] += batch_metrics[key]
            num_batches += 1
            
            # Update loss scheduler
            self.loss_scheduler.step(batch_metrics)
            
            if batch_idx % 10 == 0:
                self.logger.debug(
                    f"Batch {batch_idx}/{len(train_dataloader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Ranking Acc: {batch_metrics.get('ranking_accuracy', 0):.4f}"
                )
        
        # Compute epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            training_loss=avg_loss,
            ranking_accuracy=total_metrics["ranking_accuracy"],
            margin_satisfaction=total_metrics["margin_satisfaction"],
            avg_quantum_fidelity=total_metrics["avg_quantum_fidelity"],
            training_time=epoch_time,
            memory_usage_gb=self._get_memory_usage()
        )
        
        return metrics
    
    def validate_epoch(
        self,
        val_dataloader: torch.utils.data.DataLoader
    ) -> TrainingMetrics:
        """Validate for one epoch."""
        self.quantum_trainer.eval()
        self.classical_trainer.eval()
        
        total_loss = 0.0
        total_metrics = {
            "ranking_accuracy": 0.0,
            "margin_satisfaction": 0.0,
            "avg_quantum_fidelity": 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for anchor, positive, negative in val_dataloader:
                # Move to device
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                loss, _, batch_metrics = self.hybrid_training_step(
                    anchor, positive, negative
                )
                
                # Accumulate metrics
                total_loss += loss.item()
                for key in total_metrics:
                    if key in batch_metrics:
                        total_metrics[key] += batch_metrics[key]
                num_batches += 1
        
        # Compute validation metrics
        avg_loss = total_loss / num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            validation_loss=avg_loss,
            ranking_accuracy=total_metrics["ranking_accuracy"],
            margin_satisfaction=total_metrics["margin_satisfaction"],
            avg_quantum_fidelity=total_metrics["avg_quantum_fidelity"]
        )
        
        return metrics
    
    def _compute_training_metrics(
        self,
        anchor_similarity: torch.Tensor,
        positive_similarity: torch.Tensor,
        negative_similarity: torch.Tensor,
        quantum_fidelities: torch.Tensor
    ) -> Dict[str, float]:
        """Compute training metrics for monitoring."""
        with torch.no_grad():
            # Ranking accuracy
            ranking_accuracy = (positive_similarity > negative_similarity).float().mean()
            
            # Margin satisfaction
            margin_satisfaction = (
                positive_similarity - negative_similarity > self.config.triplet_margin
            ).float().mean()
            
            # Average quantum fidelity
            avg_quantum_fidelity = quantum_fidelities.mean()
            
            return {
                "ranking_accuracy": ranking_accuracy.item(),
                "margin_satisfaction": margin_satisfaction.item(),
                "avg_quantum_fidelity": avg_quantum_fidelity.item()
            }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        else:
            # Use psutil for CPU memory if available
            try:
                import psutil
                return psutil.virtual_memory().used / 1024**3
            except ImportError:
                return 0.0
    
    def _should_early_stop(self, validation_loss: float) -> bool:
        """Check if training should stop early."""
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def save_checkpoint(self, filepath: Path) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "quantum_trainer_state": self.quantum_trainer.state_dict(),
            "classical_trainer_state": self.classical_trainer.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "training_history": [m.to_dict() for m in self.training_history],
            "config": self.config.__dict__
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_epoch = checkpoint["epoch"]
        self.quantum_trainer.load_state_dict(checkpoint["quantum_trainer_state"])
        self.classical_trainer.load_state_dict(checkpoint["classical_trainer_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_validation_loss = checkpoint["best_validation_loss"]
        
        # Restore training history
        self.training_history = [
            TrainingMetrics(**m) for m in checkpoint["training_history"]
        ]
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
    
    def configure(self, config: QuantumRerankConfigSchema) -> None:
        """Configure component with new configuration."""
        # Update training configuration from global config
        if hasattr(config, 'training'):
            training_config = config.training
            # Update relevant parameters
            self.config.batch_size = getattr(training_config, 'batch_size', self.config.batch_size)
            self.config.max_epochs = getattr(training_config, 'max_epochs', self.config.max_epochs)
    
    def validate_config(self, config: QuantumRerankConfigSchema) -> bool:
        """Validate configuration for this component."""
        # Check if quantum configuration is compatible
        if hasattr(config, 'quantum'):
            quantum_config = config.quantum
            if quantum_config.n_qubits != self.config.n_qubits:
                return False
            if quantum_config.max_circuit_depth != self.config.circuit_depth:
                return False
        return True
    
    def get_config_requirements(self) -> List[str]:
        """Get list of required configuration sections."""
        return ["quantum", "ml", "performance"]


__all__ = [
    "HybridTrainingConfig",
    "TrainingMetrics",
    "QuantumCircuitTrainer",
    "ClassicalMLPTrainer", 
    "HybridQuantumClassicalTrainer"
]