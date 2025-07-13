"""
Training Pipeline for Quantum Parameter Prediction.

This module implements the training pipeline for quantum parameter predictors
using fidelity-based triplet loss for hybrid quantum-classical optimization.

Based on:
- PRD Section 3.1: Core Algorithms - Hybrid Training
- PennyLane documentation for quantum-classical training
- Research papers: Quantum fidelity triplet loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
from dataclasses import dataclass

from .parameter_predictor import QuantumParameterPredictor, ParameterPredictorConfig
from .parameterized_circuits import ParameterizedQuantumCircuits

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for parameter predictor training."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    validation_split: float = 0.2
    patience: int = 10  # Early stopping patience
    min_delta: float = 1e-4  # Minimum improvement for early stopping
    weight_decay: float = 1e-5
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    gradient_clip_norm: float = 1.0
    triplet_margin: float = 0.3
    use_quantum_fidelity: bool = True  # Use quantum fidelity or parameter similarity
    fidelity_weight: float = 0.7  # Weight for quantum fidelity loss
    parameter_weight: float = 0.3  # Weight for parameter similarity loss


class FidelityTripletLoss(nn.Module):
    """
    Triplet loss using quantum fidelity as similarity metric.
    
    Based on research papers and PRD hybrid training approach.
    The loss encourages higher fidelity between anchor-positive pairs
    and lower fidelity between anchor-negative pairs.
    """
    
    def __init__(self, margin: float = 0.3, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.margin = margin
        self.circuit_builder = ParameterizedQuantumCircuits(n_qubits, n_layers)
    
    def forward(self, 
                anchor_params: Dict[str, torch.Tensor],
                positive_params: Dict[str, torch.Tensor],
                negative_params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute triplet loss using quantum fidelity.
        
        Args:
            anchor_params: Quantum parameters for anchor
            positive_params: Quantum parameters for positive
            negative_params: Quantum parameters for negative
            
        Returns:
            Tuple of (loss, metrics)
        """
        batch_size = anchor_params['ry_params'].shape[0]
        device = anchor_params['ry_params'].device
        
        fidelities_pos = []
        fidelities_neg = []
        
        # Compute fidelities for each sample in batch
        for i in range(batch_size):
            # Extract parameters for this sample
            anchor_sample = {k: v[i:i+1] for k, v in anchor_params.items()}
            positive_sample = {k: v[i:i+1] for k, v in positive_params.items()}
            negative_sample = {k: v[i:i+1] for k, v in negative_params.items()}
            
            try:
                # Create quantum circuits
                anchor_circuit = self.circuit_builder.create_parameterized_circuit(anchor_sample, 0)
                positive_circuit = self.circuit_builder.create_parameterized_circuit(positive_sample, 0)
                negative_circuit = self.circuit_builder.create_parameterized_circuit(negative_sample, 0)
                
                # Compute fidelities
                fidelity_pos, _ = self.circuit_builder.compute_circuit_fidelity(
                    anchor_circuit, positive_circuit
                )
                fidelity_neg, _ = self.circuit_builder.compute_circuit_fidelity(
                    anchor_circuit, negative_circuit
                )
                
                fidelities_pos.append(fidelity_pos)
                fidelities_neg.append(fidelity_neg)
                
            except Exception as e:
                logger.warning(f"Fidelity computation failed for sample {i}: {e}")
                # Use fallback similarity based on parameter distance
                anchor_flat = torch.cat([v[i] for v in anchor_params.values()])
                positive_flat = torch.cat([v[i] for v in positive_params.values()])
                negative_flat = torch.cat([v[i] for v in negative_params.values()])
                
                # Cosine similarity as fallback
                fidelity_pos = F.cosine_similarity(
                    anchor_flat.unsqueeze(0), positive_flat.unsqueeze(0)
                ).item()
                fidelity_neg = F.cosine_similarity(
                    anchor_flat.unsqueeze(0), negative_flat.unsqueeze(0)
                ).item()
                
                # Scale to [0, 1]
                fidelity_pos = (fidelity_pos + 1) / 2
                fidelity_neg = (fidelity_neg + 1) / 2
                
                fidelities_pos.append(fidelity_pos)
                fidelities_neg.append(fidelity_neg)
        
        # Convert to tensors
        fidelities_pos = torch.tensor(fidelities_pos, device=device, dtype=torch.float32)
        fidelities_neg = torch.tensor(fidelities_neg, device=device, dtype=torch.float32)
        
        # Triplet loss: maximize positive fidelity, minimize negative fidelity
        loss = torch.clamp(self.margin - fidelities_pos + fidelities_neg, min=0.0)
        
        # Compute metrics
        metrics = {
            'avg_fidelity_positive': fidelities_pos.mean().item(),
            'avg_fidelity_negative': fidelities_neg.mean().item(),
            'fidelity_margin': (fidelities_pos - fidelities_neg).mean().item(),
            'triplet_loss': loss.mean().item()
        }
        
        return loss.mean(), metrics


class ParameterSimilarityTripletLoss(nn.Module):
    """
    Faster triplet loss using parameter space similarity instead of quantum fidelity.
    
    Used as an alternative when quantum fidelity computation is too expensive
    or as a component in hybrid loss functions.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self,
                anchor_params: Dict[str, torch.Tensor],
                positive_params: Dict[str, torch.Tensor], 
                negative_params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute triplet loss using parameter space cosine similarity.
        
        Args:
            anchor_params: Parameters for anchor
            positive_params: Parameters for positive  
            negative_params: Parameters for negative
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Flatten all parameter types
        anchor_flat = torch.cat([v for v in anchor_params.values()], dim=1)
        positive_flat = torch.cat([v for v in positive_params.values()], dim=1)
        negative_flat = torch.cat([v for v in negative_params.values()], dim=1)
        
        # Compute cosine similarities
        similarity_pos = F.cosine_similarity(anchor_flat, positive_flat, dim=1)
        similarity_neg = F.cosine_similarity(anchor_flat, negative_flat, dim=1)
        
        # Scale to [0, 1] range
        similarity_pos = (similarity_pos + 1) / 2
        similarity_neg = (similarity_neg + 1) / 2
        
        # Triplet loss
        loss = torch.clamp(self.margin - similarity_pos + similarity_neg, min=0.0)
        
        # Compute metrics
        metrics = {
            'avg_similarity_positive': similarity_pos.mean().item(),
            'avg_similarity_negative': similarity_neg.mean().item(), 
            'similarity_margin': (similarity_pos - similarity_neg).mean().item(),
            'parameter_loss': loss.mean().item()
        }
        
        return loss.mean(), metrics


class HybridTripletLoss(nn.Module):
    """
    Hybrid loss combining quantum fidelity and parameter similarity.
    
    Provides a balance between quantum-aware training and computational efficiency.
    """
    
    def __init__(self, margin: float = 0.3, n_qubits: int = 4, n_layers: int = 2,
                 fidelity_weight: float = 0.7, parameter_weight: float = 0.3):
        super().__init__()
        self.fidelity_loss = FidelityTripletLoss(margin, n_qubits, n_layers)
        self.parameter_loss = ParameterSimilarityTripletLoss(margin)
        self.fidelity_weight = fidelity_weight
        self.parameter_weight = parameter_weight
    
    def forward(self,
                anchor_params: Dict[str, torch.Tensor],
                positive_params: Dict[str, torch.Tensor],
                negative_params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid triplet loss.
        
        Returns:
            Tuple of (loss, metrics)
        """
        # Compute both loss components
        fidelity_loss, fidelity_metrics = self.fidelity_loss(
            anchor_params, positive_params, negative_params
        )
        parameter_loss, parameter_metrics = self.parameter_loss(
            anchor_params, positive_params, negative_params
        )
        
        # Weighted combination
        total_loss = (self.fidelity_weight * fidelity_loss + 
                     self.parameter_weight * parameter_loss)
        
        # Combined metrics
        metrics = {
            'total_loss': total_loss.item(),
            'fidelity_loss': fidelity_loss.item(),
            'parameter_loss': parameter_loss.item(),
            **{f"fidelity_{k}": v for k, v in fidelity_metrics.items()},
            **{f"parameter_{k}": v for k, v in parameter_metrics.items()}
        }
        
        return total_loss, metrics


class ParameterPredictorTrainer:
    """
    Trainer for quantum parameter predictor using fidelity-based loss.
    
    Supports different loss functions and training strategies for hybrid
    quantum-classical optimization.
    """
    
    def __init__(self, 
                 model: QuantumParameterPredictor,
                 config: TrainingConfig = None):
        self.model = model
        self.config = config or TrainingConfig()
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )
        
        # Setup loss function
        self._setup_loss_function()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'metrics': []
        }
        
        logger.info("ParameterPredictorTrainer initialized")
    
    def _setup_loss_function(self):
        """Setup the appropriate loss function based on configuration."""
        n_qubits = self.model.config.n_qubits
        n_layers = self.model.config.n_layers
        margin = self.config.triplet_margin
        
        if self.config.use_quantum_fidelity:
            if self.config.fidelity_weight < 1.0:
                # Use hybrid loss
                self.loss_function = HybridTripletLoss(
                    margin=margin,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    fidelity_weight=self.config.fidelity_weight,
                    parameter_weight=self.config.parameter_weight
                )
                logger.info("Using hybrid fidelity + parameter triplet loss")
            else:
                # Use pure fidelity loss
                self.loss_function = FidelityTripletLoss(margin, n_qubits, n_layers)
                logger.info("Using quantum fidelity triplet loss")
        else:
            # Use parameter similarity loss
            self.loss_function = ParameterSimilarityTripletLoss(margin)
            logger.info("Using parameter similarity triplet loss")
    
    def create_triplet_dataset(self, 
                             embeddings: np.ndarray,
                             similarity_labels: Optional[np.ndarray] = None) -> TensorDataset:
        """
        Create triplet dataset from embeddings.
        
        Args:
            embeddings: Array of embeddings [n_samples, embedding_dim]
            similarity_labels: Optional similarity matrix for supervised triplets
            
        Returns:
            TensorDataset with (anchor, positive, negative) triplets
        """
        n_samples = len(embeddings)
        
        # Generate triplets
        anchors, positives, negatives = [], [], []
        
        for i in range(n_samples):
            # Anchor
            anchor = embeddings[i]
            
            # Find positive (similar sample)
            if similarity_labels is not None:
                # Use similarity labels if available
                positive_candidates = np.where(similarity_labels[i] > 0.7)[0]
                positive_candidates = positive_candidates[positive_candidates != i]  # Exclude self
                if len(positive_candidates) > 0:
                    pos_idx = np.random.choice(positive_candidates)
                else:
                    pos_idx = np.random.choice([j for j in range(n_samples) if j != i])
            else:
                # Random positive (could be improved with actual similarity computation)
                pos_idx = np.random.choice([j for j in range(n_samples) if j != i])
            
            positive = embeddings[pos_idx]
            
            # Find negative (dissimilar sample)
            if similarity_labels is not None:
                negative_candidates = np.where(similarity_labels[i] < 0.3)[0]
                negative_candidates = negative_candidates[negative_candidates != i]  # Exclude self
                if len(negative_candidates) > 0:
                    neg_idx = np.random.choice(negative_candidates)
                else:
                    neg_idx = np.random.choice([j for j in range(n_samples) if j != i])
            else:
                # Random negative
                neg_idx = np.random.choice([j for j in range(n_samples) if j != i])
            
            negative = embeddings[neg_idx]
            
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
        
        # Convert to tensors
        anchor_tensor = torch.FloatTensor(np.array(anchors))
        positive_tensor = torch.FloatTensor(np.array(positives))
        negative_tensor = torch.FloatTensor(np.array(negatives))
        
        return TensorDataset(anchor_tensor, positive_tensor, negative_tensor)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        epoch_metrics = {}
        
        for batch_idx, (anchors, positives, negatives) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Predict parameters for all triplet components
            anchor_params = self.model(anchors)
            positive_params = self.model(positives)
            negative_params = self.model(negatives)
            
            # Compute loss
            loss, metrics = self.loss_function(anchor_params, positive_params, negative_params)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
        
        # Average metrics
        avg_loss = total_loss / len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)
        
        return avg_loss, epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        epoch_metrics = {}
        
        with torch.no_grad():
            for anchors, positives, negatives in val_loader:
                # Predict parameters
                anchor_params = self.model(anchors)
                positive_params = self.model(positives)
                negative_params = self.model(negatives)
                
                # Compute loss
                loss, metrics = self.loss_function(anchor_params, positive_params, negative_params)
                
                total_loss += loss.item()
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value
        
        # Average metrics
        avg_loss = total_loss / len(val_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= len(val_loader)
        
        return avg_loss, epoch_metrics
    
    def train(self, dataset: TensorDataset) -> Dict:
        """
        Train the parameter predictor.
        
        Args:
            dataset: Triplet dataset for training
            
        Returns:
            Training history and final metrics
        """
        logger.info(f"Starting training: {self.config.num_epochs} epochs, {len(dataset)} samples")
        
        # Split dataset
        dataset_size = len(dataset)
        val_size = int(self.config.validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['metrics'].append({
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
            
            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                if 'fidelity_margin' in train_metrics:
                    logger.info(f"  Fidelity margin: {train_metrics['fidelity_margin']:.4f}")
        
        training_results = {
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.history['train_loss']),
            'history': self.history,
            'config': self.config
        }
        
        logger.info("Training completed")
        return training_results
    
    def save_checkpoint(self, filepath: str, epoch: int, additional_info: Dict = None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config,
            'model_config': self.model.config
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint