"""
Hybrid Quantum-Classical Training Pipeline for QuantumRerank.

This module implements hybrid training combining quantum parameterized circuits
with classical neural networks for optimized similarity computation and ranking.
"""

from .hybrid_trainer import HybridQuantumClassicalTrainer
from .loss_functions import (
    QuantumTripletLoss, RankingLoss, ContrastiveLoss,
    quantum_triplet_loss, ranking_loss, contrastive_loss
)
from .optimizers import (
    QuantumOptimizer, HybridOptimizer, 
    create_quantum_optimizer, create_hybrid_optimizer
)
from .data_manager import (
    TrainingDataManager, TripletDataset,
    HardNegativeMiner, DataAugmentator
)
from .trainer import (
    TrainingLoop, TrainingConfig, TrainingMetrics,
    ValidationMetrics, ModelCheckpoint
)

__all__ = [
    # Main hybrid trainer
    "HybridQuantumClassicalTrainer",
    
    # Loss functions
    "QuantumTripletLoss", "RankingLoss", "ContrastiveLoss",
    "quantum_triplet_loss", "ranking_loss", "contrastive_loss",
    
    # Optimizers
    "QuantumOptimizer", "HybridOptimizer",
    "create_quantum_optimizer", "create_hybrid_optimizer",
    
    # Data management
    "TrainingDataManager", "TripletDataset",
    "HardNegativeMiner", "DataAugmentator",
    
    # Training loop
    "TrainingLoop", "TrainingConfig", "TrainingMetrics",
    "ValidationMetrics", "ModelCheckpoint"
]