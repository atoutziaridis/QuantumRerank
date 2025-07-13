"""
Machine Learning Module for Quantum Parameter Prediction.

This module implements classical ML components for hybrid quantum-classical
training, including parameter prediction, parameterized circuit creation,
and training pipelines.

Based on Task 05: Quantum Parameter Prediction with Classical MLP.
"""

from .parameter_predictor import (
    QuantumParameterPredictor,
    ParameterPredictorConfig
)

from .parameterized_circuits import (
    ParameterizedQuantumCircuits
)

from .training import (
    ParameterPredictorTrainer,
    TrainingConfig,
    FidelityTripletLoss,
    ParameterSimilarityTripletLoss,
    HybridTripletLoss
)

from .parameter_integration import (
    EmbeddingToCircuitPipeline
)

__all__ = [
    'QuantumParameterPredictor',
    'ParameterPredictorConfig',
    'ParameterizedQuantumCircuits', 
    'ParameterPredictorTrainer',
    'TrainingConfig',
    'FidelityTripletLoss',
    'ParameterSimilarityTripletLoss',
    'HybridTripletLoss',
    'EmbeddingToCircuitPipeline'
]