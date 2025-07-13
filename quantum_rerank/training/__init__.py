"""
Hybrid Quantum-Classical Training Pipeline for QuantumRerank.

This module implements hybrid training combining quantum parameterized circuits
with classical neural networks for optimized similarity computation and ranking.
"""

# QPMeL training components
try:
    from .qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
    from .triplet_generator import TripletGenerator, TripletGeneratorConfig
except ImportError as e:
    pass  # QPMeL components may not be available

# Original training components (optional imports to avoid breaking existing code)
try:
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
except ImportError as e:
    pass  # Some components may not be available

__all__ = [
    # QPMeL components
    "QPMeLTrainer", "QPMeLTrainingConfig",
    "TripletGenerator", "TripletGeneratorConfig",
    
    # Main hybrid trainer (if available)
    "HybridQuantumClassicalTrainer",
    
    # Loss functions (if available)
    "QuantumTripletLoss", "RankingLoss", "ContrastiveLoss",
    "quantum_triplet_loss", "ranking_loss", "contrastive_loss",
    
    # Optimizers (if available)
    "QuantumOptimizer", "HybridOptimizer",
    "create_quantum_optimizer", "create_hybrid_optimizer",
    
    # Data management (if available)
    "TrainingDataManager", "TripletDataset",
    "HardNegativeMiner", "DataAugmentator",
    
    # Training loop (if available)
    "TrainingLoop", "TrainingConfig", "TrainingMetrics",
    "ValidationMetrics", "ModelCheckpoint"
]