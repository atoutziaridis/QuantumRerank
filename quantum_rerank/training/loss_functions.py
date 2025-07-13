"""
Loss functions for hybrid quantum-classical training.

This module provides ranking-optimized loss functions including triplet loss,
margin-based ranking loss, and quantum-aware loss functions for similarity learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..utils import get_logger


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    margin: float = 0.5
    temperature: float = 1.0
    reduction: str = "mean"  # "mean", "sum", "none"
    use_hard_negative: bool = True
    negative_weight: float = 1.0
    positive_weight: float = 1.0


class QuantumTripletLoss(nn.Module):
    """
    Quantum-enhanced triplet loss for similarity learning.
    
    This loss function is specifically designed for ranking tasks with
    quantum similarity computation, optimizing for ranking accuracy improvement.
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        reduction: str = "mean",
        use_hard_negative: bool = True
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.use_hard_negative = use_hard_negative
        self.logger = get_logger(__name__)
    
    def forward(
        self,
        anchor_similarity: torch.Tensor,
        positive_similarity: torch.Tensor,
        negative_similarity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantum triplet loss.
        
        Args:
            anchor_similarity: Similarity between anchor and itself (should be ~1.0)
            positive_similarity: Similarity between anchor and positive samples
            negative_similarity: Similarity between anchor and negative samples
            
        Returns:
            Computed triplet loss
        """
        # Triplet loss: max(0, margin + negative_similarity - positive_similarity)
        loss = torch.clamp(
            self.margin + negative_similarity - positive_similarity,
            min=0.0
        )
        
        if self.use_hard_negative:
            # Focus on hard negatives (high similarity with negatives)
            hard_negative_mask = negative_similarity > negative_similarity.median()
            if hard_negative_mask.sum() > 0:
                loss = loss * hard_negative_mask.float()
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def compute_metrics(
        self,
        anchor_similarity: torch.Tensor,
        positive_similarity: torch.Tensor, 
        negative_similarity: torch.Tensor
    ) -> Dict[str, float]:
        """Compute training metrics for monitoring."""
        with torch.no_grad():
            # Ranking accuracy: fraction of cases where positive > negative
            ranking_accuracy = (positive_similarity > negative_similarity).float().mean()
            
            # Margin satisfaction: fraction of cases satisfying the margin
            margin_satisfaction = (
                positive_similarity - negative_similarity > self.margin
            ).float().mean()
            
            # Average similarities
            avg_positive_sim = positive_similarity.mean()
            avg_negative_sim = negative_similarity.mean()
            
            return {
                "ranking_accuracy": ranking_accuracy.item(),
                "margin_satisfaction": margin_satisfaction.item(),
                "avg_positive_similarity": avg_positive_sim.item(),
                "avg_negative_similarity": avg_negative_sim.item(),
                "similarity_gap": (avg_positive_sim - avg_negative_sim).item()
            }


class RankingLoss(nn.Module):
    """Margin-based ranking loss for similarity learning."""
    
    def __init__(
        self,
        margin: float = 0.5,
        reduction: str = "mean",
        temperature: float = 1.0
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.temperature = temperature
    
    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ranking loss with temperature scaling.
        
        Args:
            positive_scores: Scores for positive pairs
            negative_scores: Scores for negative pairs
            
        Returns:
            Computed ranking loss
        """
        # Apply temperature scaling
        positive_scores = positive_scores / self.temperature
        negative_scores = negative_scores / self.temperature
        
        # Ranking loss with margin
        loss = torch.clamp(
            self.margin - positive_scores + negative_scores,
            min=0.0
        )
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for semantic similarity learning."""
    
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        similarity_scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            similarity_scores: Computed similarity scores
            labels: Binary labels (1 for similar pairs, 0 for dissimilar)
            
        Returns:
            Computed contrastive loss
        """
        # Convert similarities to distances (1 - similarity)
        distances = 1.0 - similarity_scores
        
        # Contrastive loss
        positive_loss = labels.float() * torch.pow(distances, 2)
        negative_loss = (1 - labels).float() * torch.pow(
            torch.clamp(self.margin - distances, min=0.0), 2
        )
        
        loss = positive_loss + negative_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class QuantumAwareLoss(nn.Module):
    """
    Quantum-aware loss function that incorporates quantum fidelity constraints.
    
    This loss function is designed to work with quantum similarity computations
    and includes regularization terms for quantum circuit parameter learning.
    """
    
    def __init__(
        self,
        triplet_weight: float = 1.0,
        fidelity_weight: float = 0.1,
        parameter_reg_weight: float = 0.01,
        margin: float = 0.5
    ):
        super().__init__()
        self.triplet_weight = triplet_weight
        self.fidelity_weight = fidelity_weight
        self.parameter_reg_weight = parameter_reg_weight
        self.triplet_loss = QuantumTripletLoss(margin=margin)
    
    def forward(
        self,
        anchor_similarity: torch.Tensor,
        positive_similarity: torch.Tensor,
        negative_similarity: torch.Tensor,
        quantum_parameters: Optional[torch.Tensor] = None,
        quantum_fidelities: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute quantum-aware composite loss.
        
        Args:
            anchor_similarity: Anchor similarities
            positive_similarity: Positive similarities  
            negative_similarity: Negative similarities
            quantum_parameters: Quantum circuit parameters
            quantum_fidelities: Quantum state fidelities
            
        Returns:
            Total loss and component losses
        """
        # Primary triplet loss
        triplet_loss = self.triplet_loss(
            anchor_similarity, positive_similarity, negative_similarity
        )
        
        total_loss = self.triplet_weight * triplet_loss
        loss_components = {"triplet_loss": triplet_loss}
        
        # Quantum fidelity regularization
        if quantum_fidelities is not None:
            # Encourage high quantum fidelity
            fidelity_loss = -torch.log(quantum_fidelities + 1e-8).mean()
            total_loss += self.fidelity_weight * fidelity_loss
            loss_components["fidelity_loss"] = fidelity_loss
        
        # Quantum parameter regularization
        if quantum_parameters is not None:
            # L2 regularization on quantum parameters
            param_reg_loss = torch.norm(quantum_parameters, p=2).mean()
            total_loss += self.parameter_reg_weight * param_reg_loss
            loss_components["parameter_reg_loss"] = param_reg_loss
        
        return total_loss, loss_components


# Functional interfaces for loss functions

def quantum_triplet_loss(
    anchor_similarity: torch.Tensor,
    positive_similarity: torch.Tensor,
    negative_similarity: torch.Tensor,
    margin: float = 0.5
) -> torch.Tensor:
    """
    Functional interface for quantum triplet loss.
    
    Args:
        anchor_similarity: Similarity between anchor and itself
        positive_similarity: Similarity between anchor and positive samples
        negative_similarity: Similarity between anchor and negative samples
        margin: Margin for triplet loss
        
    Returns:
        Computed triplet loss
    """
    loss_fn = QuantumTripletLoss(margin=margin)
    return loss_fn(anchor_similarity, positive_similarity, negative_similarity)


def ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float = 0.5,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Functional interface for ranking loss.
    
    Args:
        positive_scores: Scores for positive pairs
        negative_scores: Scores for negative pairs
        margin: Margin for ranking loss
        temperature: Temperature for score scaling
        
    Returns:
        Computed ranking loss
    """
    loss_fn = RankingLoss(margin=margin, temperature=temperature)
    return loss_fn(positive_scores, negative_scores)


def contrastive_loss(
    similarity_scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """
    Functional interface for contrastive loss.
    
    Args:
        similarity_scores: Computed similarity scores
        labels: Binary labels for pairs
        margin: Margin for contrastive loss
        
    Returns:
        Computed contrastive loss
    """
    loss_fn = ContrastiveLoss(margin=margin)
    return loss_fn(similarity_scores, labels)


class AdaptiveLossScheduler:
    """
    Adaptive loss scheduling for dynamic loss weight adjustment during training.
    
    This scheduler adjusts loss weights based on training progress and performance
    to ensure balanced learning across different loss components.
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_schedule: str = "cosine",  # "linear", "cosine", "step"
        adaptation_steps: int = 1000
    ):
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.adaptation_schedule = adaptation_schedule
        self.adaptation_steps = adaptation_steps
        self.step_count = 0
    
    def step(self, metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Update loss weights based on training progress."""
        self.step_count += 1
        
        if self.adaptation_schedule == "cosine":
            progress = min(self.step_count / self.adaptation_steps, 1.0)
            decay_factor = 0.5 * (1 + np.cos(np.pi * progress))
            
            # Gradually reduce auxiliary losses, maintain primary loss
            for key in self.current_weights:
                if key != "triplet_loss":
                    self.current_weights[key] = (
                        self.initial_weights[key] * decay_factor
                    )
        
        elif self.adaptation_schedule == "linear":
            progress = min(self.step_count / self.adaptation_steps, 1.0)
            decay_factor = 1.0 - progress
            
            for key in self.current_weights:
                if key != "triplet_loss":
                    self.current_weights[key] = (
                        self.initial_weights[key] * decay_factor
                    )
        
        # Adaptive adjustment based on metrics
        if metrics:
            self._adapt_to_metrics(metrics)
        
        return self.current_weights.copy()
    
    def _adapt_to_metrics(self, metrics: Dict[str, float]) -> None:
        """Adapt weights based on training metrics."""
        # Increase fidelity weight if quantum fidelity is low
        if "avg_quantum_fidelity" in metrics:
            fidelity = metrics["avg_quantum_fidelity"]
            if fidelity < 0.8 and "fidelity_loss" in self.current_weights:
                self.current_weights["fidelity_loss"] *= 1.1
        
        # Adjust regularization based on overfitting indicators
        if "validation_loss" in metrics and "training_loss" in metrics:
            val_loss = metrics["validation_loss"]
            train_loss = metrics["training_loss"]
            if val_loss > train_loss * 1.2:  # Overfitting indicator
                if "parameter_reg_loss" in self.current_weights:
                    self.current_weights["parameter_reg_loss"] *= 1.1


__all__ = [
    "LossConfig",
    "QuantumTripletLoss",
    "RankingLoss", 
    "ContrastiveLoss",
    "QuantumAwareLoss",
    "AdaptiveLossScheduler",
    "quantum_triplet_loss",
    "ranking_loss",
    "contrastive_loss"
]