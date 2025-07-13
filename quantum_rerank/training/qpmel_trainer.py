"""
QPMeL Fidelity Triplet Loss Training Implementation.

Implements the core training loop for Quantum Polar Metric Learning using
quantum fidelity as the similarity metric in triplet loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..core.embeddings import EmbeddingProcessor
from ..core.swap_test import QuantumSWAPTest
from ..ml.qpmel_circuits import QPMeLParameterPredictor, QPMeLConfig
from ..config.settings import QuantumConfig

logger = logging.getLogger(__name__)

@dataclass
class QPMeLTrainingConfig:
    """Configuration for QPMeL training."""
    # Model parameters
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Training parameters
    batch_size: int = 32
    max_epochs: int = 100
    triplet_margin: float = 0.2
    
    # Quantum parameters
    qpmel_config: QPMeLConfig = field(default_factory=QPMeLConfig)
    
    # Training control
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    save_best_model: bool = True
    
    # Logging
    log_interval: int = 10
    validate_interval: int = 5

class TripletDataset(Dataset):
    """Dataset for triplet training with (anchor, positive, negative) samples."""
    
    def __init__(self, triplets: List[Tuple[str, str, str]]):
        """
        Initialize triplet dataset.
        
        Args:
            triplets: List of (anchor, positive, negative) text triplets
        """
        self.triplets = triplets
        
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]

class FidelityTripletLoss(nn.Module):
    """
    Quantum Fidelity Triplet Loss as described in QPMeL paper.
    
    Loss = max(fidelity(anchor, negative) - fidelity(anchor, positive) + margin, 0)
    
    This pushes positive pairs to have higher quantum fidelity than negative pairs.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, 
                fidelity_pos: torch.Tensor, 
                fidelity_neg: torch.Tensor) -> torch.Tensor:
        """
        Compute fidelity triplet loss.
        
        Args:
            fidelity_pos: Fidelity between anchor and positive (batch_size,)
            fidelity_neg: Fidelity between anchor and negative (batch_size,)
            
        Returns:
            Triplet loss tensor
        """
        # Triplet loss: want fidelity_pos > fidelity_neg + margin
        loss = torch.clamp(fidelity_neg - fidelity_pos + self.margin, min=0.0)
        return loss.mean()

class QPMeLTrainer:
    """
    Trainer for QPMeL (Quantum Polar Metric Learning) using fidelity triplet loss.
    
    This implements the core training loop that makes quantum circuit parameters
    meaningful for semantic similarity.
    """
    
    def __init__(self, 
                 config: QPMeLTrainingConfig,
                 embedding_processor: Optional[EmbeddingProcessor] = None,
                 device: Optional[torch.device] = None):
        
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize embedding processor
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        embedding_dim = self.embedding_processor.model.get_sentence_embedding_dimension()
        
        # Initialize quantum components
        self.swap_test = QuantumSWAPTest(
            n_qubits=config.qpmel_config.n_qubits,
            config=QuantumConfig(n_qubits=config.qpmel_config.n_qubits)
        )
        
        # Initialize model
        self.model = QPMeLParameterPredictor(
            input_dim=embedding_dim,
            config=config.qpmel_config,
            hidden_dims=config.hidden_dims
        ).to(self.device)
        
        # Initialize loss and optimizer
        self.criterion = FidelityTripletLoss(margin=config.triplet_margin)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        logger.info(f"QPMeL trainer initialized on {self.device}")
        logger.info(f"Model: {self.model.get_model_info()}")
    
    def prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of triplets for training.
        
        Args:
            batch: Batch from DataLoader - can be list of triplets or tensor
            
        Returns:
            Tuple of (anchor_embeddings, positive_embeddings, negative_embeddings)
        """
        # Handle different batch formats from DataLoader
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 3:
                # Standard format: list of triplets
                anchors, positives, negatives = zip(*batch)
            else:
                # DataLoader might return a single batch item
                anchors, positives, negatives = batch
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}, content: {batch}")
        
        # Encode all texts in batch
        all_texts = list(anchors) + list(positives) + list(negatives)
        all_embeddings = self.embedding_processor.encode_texts(all_texts)
        
        batch_size = len(anchors)
        
        # Split embeddings
        anchor_embeddings = all_embeddings[:batch_size]
        positive_embeddings = all_embeddings[batch_size:2*batch_size]
        negative_embeddings = all_embeddings[2*batch_size:]
        
        # Convert to tensors
        anchor_tensor = torch.FloatTensor(anchor_embeddings).to(self.device)
        positive_tensor = torch.FloatTensor(positive_embeddings).to(self.device)
        negative_tensor = torch.FloatTensor(negative_embeddings).to(self.device)
        
        return anchor_tensor, positive_tensor, negative_tensor
    
    def compute_batch_fidelities(self, 
                                embeddings1: torch.Tensor,
                                embeddings2: torch.Tensor,
                                training: bool = True) -> torch.Tensor:
        """
        Compute differentiable quantum fidelities between two batches of embeddings.
        
        For training, we use a differentiable approximation based on parameter similarity.
        This maintains gradients while approximating quantum fidelity behavior.
        
        Args:
            embeddings1: First batch of embeddings
            embeddings2: Second batch of embeddings
            training: Whether in training mode
            
        Returns:
            Tensor of fidelities with gradients preserved
        """
        # Get quantum circuit parameters (these have gradients)
        params1 = self.model.forward(embeddings1, training=training)
        params2 = self.model.forward(embeddings2, training=training)
        
        if training:
            # Differentiable approximation: normalized cosine similarity of parameters
            # This preserves gradients and approximates quantum fidelity behavior
            params1_norm = torch.nn.functional.normalize(params1, p=2, dim=1)
            params2_norm = torch.nn.functional.normalize(params2, p=2, dim=1)
            
            # Cosine similarity in parameter space
            cosine_sim = torch.sum(params1_norm * params2_norm, dim=1)
            
            # Map to [0, 1] and add small quantum-inspired non-linearity
            fidelities = 0.5 * (1 + cosine_sim)
            fidelities = fidelities ** 2  # Quantum fidelity is typically squared overlap
            
            return fidelities
        else:
            # For validation/evaluation, use actual quantum SWAP test
            with torch.no_grad():
                circuits1 = self.model.get_circuits(embeddings1, training=False)
                circuits2 = self.model.get_circuits(embeddings2, training=False)
                
                fidelities = []
                for i in range(len(circuits1)):
                    try:
                        fidelity, _ = self.swap_test.compute_fidelity(circuits1[i], circuits2[i])
                        fidelities.append(fidelity)
                    except Exception as e:
                        logger.warning(f"Fidelity computation failed for sample {i}: {e}")
                        fidelities.append(0.0)
                
                return torch.FloatTensor(fidelities).to(self.device)
    
    def train_step(self, batch: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Batch of triplets
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        start_time = time.time()
        
        # Prepare batch
        anchor_emb, positive_emb, negative_emb = self.prepare_batch(batch)
        
        # Compute fidelities
        fidelity_pos = self.compute_batch_fidelities(anchor_emb, positive_emb, training=True)
        fidelity_neg = self.compute_batch_fidelities(anchor_emb, negative_emb, training=True)
        
        # Compute loss
        loss = self.criterion(fidelity_pos, fidelity_neg)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute metrics
        batch_time = time.time() - start_time
        
        with torch.no_grad():
            # Accuracy: percentage of triplets where fidelity_pos > fidelity_neg
            correct = (fidelity_pos > fidelity_neg).float().mean().item()
            
            metrics = {
                'loss': loss.item(),
                'accuracy': correct,
                'fidelity_pos_mean': fidelity_pos.mean().item(),
                'fidelity_neg_mean': fidelity_neg.mean().item(),
                'fidelity_gap': (fidelity_pos - fidelity_neg).mean().item(),
                'batch_time': batch_time
            }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_fidelity_pos = 0.0
        total_fidelity_neg = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                anchor_emb, positive_emb, negative_emb = self.prepare_batch(batch)
                
                # Compute fidelities (no training mode for validation)
                fidelity_pos = self.compute_batch_fidelities(anchor_emb, positive_emb, training=False)
                fidelity_neg = self.compute_batch_fidelities(anchor_emb, negative_emb, training=False)
                
                # Compute loss
                loss = self.criterion(fidelity_pos, fidelity_neg)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += (fidelity_pos > fidelity_neg).float().mean().item()
                total_fidelity_pos += fidelity_pos.mean().item()
                total_fidelity_neg += fidelity_neg.mean().item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches,
            'val_fidelity_pos_mean': total_fidelity_pos / num_batches,
            'val_fidelity_neg_mean': total_fidelity_neg / num_batches,
            'val_fidelity_gap': (total_fidelity_pos - total_fidelity_neg) / num_batches
        }
    
    def train(self, 
              triplets: List[Tuple[str, str, str]],
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the QPMeL model using triplet loss.
        
        Args:
            triplets: List of (anchor, positive, negative) text triplets
            save_path: Path to save the trained model
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting QPMeL training with {len(triplets)} triplets")
        
        # Split data
        val_size = int(len(triplets) * self.config.validation_split)
        train_triplets = triplets[:-val_size] if val_size > 0 else triplets
        val_triplets = triplets[-val_size:] if val_size > 0 else triplets[:100]  # Small val set
        
        # Custom collate function to handle triplets properly
        def triplet_collate_fn(batch):
            """Custom collate function for triplet data."""
            return batch  # Return list of triplets as-is
        
        # Create data loaders
        train_dataset = TripletDataset(train_triplets)
        val_dataset = TripletDataset(val_triplets)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=triplet_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=triplet_collate_fn)
        
        logger.info(f"Training set: {len(train_triplets)} triplets, "
                   f"Validation set: {len(val_triplets)} triplets")
        
        self.training_history = []
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Training phase
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': 0.0,
                'train_accuracy': 0.0,
                'train_fidelity_gap': 0.0
            }
            
            for batch_idx, batch in enumerate(train_loader):
                batch_metrics = self.train_step(batch)
                
                # Accumulate metrics
                for key in ['loss', 'accuracy', 'fidelity_gap']:
                    epoch_metrics[f'train_{key}'] += batch_metrics[key]
                
                # Log batch progress
                if batch_idx % self.config.log_interval == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                               f"Loss={batch_metrics['loss']:.4f}, "
                               f"Acc={batch_metrics['accuracy']:.3f}, "
                               f"Gap={batch_metrics['fidelity_gap']:.4f}")
            
            # Average training metrics
            num_batches = len(train_loader)
            for key in ['train_loss', 'train_accuracy', 'train_fidelity_gap']:
                epoch_metrics[key] /= num_batches
            
            # Validation phase
            if epoch % self.config.validate_interval == 0:
                val_metrics = self.validate(val_loader)
                epoch_metrics.update(val_metrics)
                
                # Early stopping check
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    
                    # Save best model
                    if self.config.save_best_model and save_path:
                        self.save_model(save_path)
                        logger.info(f"Saved best model to {save_path}")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            epoch_time = time.time() - epoch_start
            epoch_metrics['epoch_time'] = epoch_time
            
            self.training_history.append(epoch_metrics)
            
            # Log epoch summary
            logger.info(f"Epoch {epoch} complete ({epoch_time:.1f}s): "
                       f"Train Loss={epoch_metrics['train_loss']:.4f}, "
                       f"Val Loss={epoch_metrics.get('val_loss', 'N/A')}")
        
        logger.info("Training complete!")
        return self.training_history
    
    def save_model(self, path: str):
        """Save the trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_info': self.model.get_model_info()
        }
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {path}")
        return checkpoint
    
    def get_trained_reranker(self):
        """
        Get a trained reranker that uses the QPMeL model.
        
        Returns:
            Configured QuantumRAGReranker with trained parameters
        """
        from ..core.rag_reranker import QuantumRAGReranker
        from ..core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
        
        # Create config for similarity engine
        engine_config = SimilarityEngineConfig(
            n_qubits=self.config.qpmel_config.n_qubits,
            n_layers=self.config.qpmel_config.n_layers,
            similarity_method=SimilarityMethod.QUANTUM_FIDELITY,
            enable_caching=True,
            performance_monitoring=True
        )
        
        # Create reranker and replace parameter predictor with trained model
        reranker = QuantumRAGReranker(config=engine_config)
        
        # Replace the parameter predictor with our trained QPMeL model
        reranker.similarity_engine.parameter_predictor = self.model
        reranker.similarity_engine.circuit_builder = self.model.circuit_builder
        
        logger.info("Created QuantumRAGReranker with trained QPMeL parameters")
        
        return reranker