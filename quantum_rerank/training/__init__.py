"""
Quantum Parameter Training Pipeline for Medical Domain.

This module implements comprehensive training pipelines for optimizing quantum
circuit parameters on medical domain data, including medical data preparation,
KTA optimization, parameter prediction, and hybrid weight optimization.

Key Components:
- Medical training data preparation from PMC corpus
- KTA optimization for quantum kernels on medical data
- Parameter predictor training on medical embeddings
- Hybrid quantum/classical weight optimization
- Complete integrated training pipeline
"""

# QRF-04 Medical Training Components
try:
    from .medical_data_preparation import (
        MedicalTrainingDataset, MedicalDataPreparationPipeline,
        MedicalTrainingConfig, TrainingPair, create_medical_training_pairs
    )
    from .quantum_kernel_trainer import (
        QuantumKernelTrainer, KTAOptimizer, QuantumKernelOptimizationPipeline,
        KTAOptimizationConfig, QuantumKernelTrainingResult
    )
    from .parameter_predictor_trainer import (
        MedicalParameterPredictorTrainer, ParameterPredictorTrainingPipeline,
        ParameterPredictorTrainingConfig, ParameterPredictorTrainingResult
    )
    from .hybrid_weight_optimizer import (
        MedicalHybridOptimizer, HybridWeightOptimizationPipeline,
        HybridWeightConfig, HybridWeightOptimizationResult
    )
    from .complete_training_pipeline import (
        CompleteQuantumTrainingPipeline, CompleteTrainingConfig,
        CompleteTrainingResult, run_complete_quantum_training
    )
except ImportError as e:
    pass  # Medical training components may not be available

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
    # QRF-04 Medical Training Components
    "MedicalTrainingDataset", "MedicalDataPreparationPipeline",
    "MedicalTrainingConfig", "TrainingPair", "create_medical_training_pairs",
    "QuantumKernelTrainer", "KTAOptimizer", "QuantumKernelOptimizationPipeline",
    "KTAOptimizationConfig", "QuantumKernelTrainingResult",
    "MedicalParameterPredictorTrainer", "ParameterPredictorTrainingPipeline",
    "ParameterPredictorTrainingConfig", "ParameterPredictorTrainingResult", 
    "MedicalHybridOptimizer", "HybridWeightOptimizationPipeline",
    "HybridWeightConfig", "HybridWeightOptimizationResult",
    "CompleteQuantumTrainingPipeline", "CompleteTrainingConfig",
    "CompleteTrainingResult", "run_complete_quantum_training",
    
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