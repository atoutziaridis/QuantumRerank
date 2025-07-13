"""
QPMeL Integration with Existing QuantumRerank System.

Provides seamless integration of trained QPMeL models with the existing
QuantumRAGReranker and QuantumSimilarityEngine infrastructure.
"""

import torch
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .quantum_similarity_engine import QuantumSimilarityEngine, SimilarityEngineConfig, SimilarityMethod
from .rag_reranker import QuantumRAGReranker
from ..training.qpmel_trainer import QPMeLTrainer
from ..ml.qpmel_circuits import QPMeLParameterPredictor, QPMeLConfig

logger = logging.getLogger(__name__)

class QPMeLIntegratedEngine(QuantumSimilarityEngine):
    """
    Quantum Similarity Engine enhanced with trained QPMeL parameters.
    
    This class extends the existing QuantumSimilarityEngine to use
    trained QPMeL parameters instead of random ones.
    """
    
    def __init__(self, 
                 config: SimilarityEngineConfig,
                 qpmel_model_path: Optional[str] = None,
                 qpmel_model: Optional[QPMeLParameterPredictor] = None):
        """
        Initialize QPMeL-enhanced similarity engine.
        
        Args:
            config: Similarity engine configuration
            qpmel_model_path: Path to trained QPMeL model file
            qpmel_model: Pre-loaded QPMeL model (alternative to path)
        """
        # Initialize parent class
        super().__init__(config)
        
        self.qpmel_model = None
        self.qpmel_config = None
        
        # Load QPMeL model if provided
        if qpmel_model_path:
            self.load_qpmel_model(qpmel_model_path)
        elif qpmel_model:
            self.qpmel_model = qpmel_model
            self.qpmel_config = qpmel_model.config
            logger.info("Using provided QPMeL model")
        
        # Replace parameter predictor if QPMeL model loaded
        if self.qpmel_model:
            self.parameter_predictor = self.qpmel_model
            self.circuit_builder = self.qpmel_model.circuit_builder
            logger.info("Replaced parameter predictor with trained QPMeL model")
    
    def load_qpmel_model(self, model_path: str):
        """Load a trained QPMeL model from file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"QPMeL model not found: {model_path}")
        
        logger.info(f"Loading QPMeL model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model configuration
        model_info = checkpoint.get('model_info', {})
        circuit_props = model_info.get('circuit_properties', {})
        
        # Create QPMeL configuration
        self.qpmel_config = QPMeLConfig(
            n_qubits=circuit_props.get('n_qubits', 4),
            n_layers=circuit_props.get('n_layers', 1),
            enable_qrc=model_info.get('enable_qrc', True),
            entangling_gate=circuit_props.get('entangling_gate', 'zz'),
            max_circuit_depth=15
        )
        
        # Create model
        input_dim = model_info.get('input_dim', 768)  # Default SentenceTransformers dim
        hidden_dims = checkpoint.get('config', {}).get('hidden_dims', [512, 256])
        
        self.qpmel_model = QPMeLParameterPredictor(
            input_dim=input_dim,
            config=self.qpmel_config,
            hidden_dims=hidden_dims
        )
        
        # Load trained weights
        self.qpmel_model.load_state_dict(checkpoint['model_state_dict'])
        self.qpmel_model.eval()
        
        logger.info(f"QPMeL model loaded successfully: {model_info}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the QPMeL integration."""
        base_info = super().get_model_info()
        
        if self.qpmel_model:
            qpmel_info = self.qpmel_model.get_model_info()
            base_info.update({
                'qpmel_integrated': True,
                'qpmel_model_info': qpmel_info,
                'qpmel_config': self.qpmel_config.__dict__ if self.qpmel_config else None
            })
        else:
            base_info['qpmel_integrated'] = False
        
        return base_info

class QPMeLReranker(QuantumRAGReranker):
    """
    QuantumRAGReranker enhanced with trained QPMeL parameters.
    
    Drop-in replacement for QuantumRAGReranker that uses trained
    semantic similarity parameters.
    """
    
    def __init__(self, 
                 config: Optional[SimilarityEngineConfig] = None,
                 qpmel_model_path: Optional[str] = None,
                 qpmel_model: Optional[QPMeLParameterPredictor] = None):
        """
        Initialize QPMeL-enhanced reranker.
        
        Args:
            config: Similarity engine configuration
            qpmel_model_path: Path to trained QPMeL model
            qpmel_model: Pre-loaded QPMeL model
        """
        # Use default config if none provided
        if config is None:
            config = SimilarityEngineConfig(
                n_qubits=4,
                n_layers=1,
                similarity_method=SimilarityMethod.QUANTUM_FIDELITY,
                enable_caching=True,
                performance_monitoring=True
            )
        
        # Initialize with QPMeL-enhanced engine
        self.qpmel_model_path = qpmel_model_path
        self.qpmel_model = qpmel_model
        
        # Initialize parent (this will create similarity_engine)
        super().__init__(config)
        
        # Replace with QPMeL-enhanced engine
        self.similarity_engine = QPMeLIntegratedEngine(
            config=config,
            qpmel_model_path=qpmel_model_path,
            qpmel_model=qpmel_model
        )
        
        logger.info("QPMeL-enhanced reranker initialized")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = super().get_model_info()
        engine_info = self.similarity_engine.get_model_info()
        
        return {
            **base_info,
            'engine_type': 'QPMeLIntegratedEngine',
            'qpmel_info': engine_info
        }

# Convenience functions for easy integration

def load_qpmel_reranker(model_path: str, 
                       config: Optional[SimilarityEngineConfig] = None) -> QPMeLReranker:
    """
    Load a QPMeL-enhanced reranker from a trained model file.
    
    Args:
        model_path: Path to trained QPMeL model
        config: Optional similarity engine configuration
        
    Returns:
        QPMeL-enhanced reranker ready for use
    """
    return QPMeLReranker(
        config=config,
        qpmel_model_path=model_path
    )

def create_qpmel_similarity_engine(model_path: str,
                                  config: Optional[SimilarityEngineConfig] = None) -> QPMeLIntegratedEngine:
    """
    Create a QPMeL-enhanced similarity engine.
    
    Args:
        model_path: Path to trained QPMeL model  
        config: Optional similarity engine configuration
        
    Returns:
        QPMeL-enhanced similarity engine
    """
    if config is None:
        config = SimilarityEngineConfig(
            similarity_method=SimilarityMethod.QUANTUM_FIDELITY,
            enable_caching=True
        )
    
    return QPMeLIntegratedEngine(
        config=config,
        qpmel_model_path=model_path
    )

def upgrade_existing_reranker(reranker: QuantumRAGReranker, 
                             model_path: str) -> QPMeLReranker:
    """
    Upgrade an existing QuantumRAGReranker with trained QPMeL parameters.
    
    Args:
        reranker: Existing QuantumRAGReranker instance
        model_path: Path to trained QPMeL model
        
    Returns:
        New QPMeL-enhanced reranker with same configuration
    """
    # Extract configuration from existing reranker
    config = reranker.config
    
    # Create new QPMeL reranker with same config
    return QPMeLReranker(
        config=config,
        qpmel_model_path=model_path
    )

# Example usage demonstration
def demonstrate_qpmel_integration():
    """Demonstrate how to use QPMeL integration."""
    
    # Example 1: Direct QPMeL reranker creation
    print("Example 1: Creating QPMeL reranker")
    try:
        qpmel_reranker = load_qpmel_reranker("models/qpmel_trained.pt")
        
        # Use it like a normal reranker
        query = "What is quantum computing?"
        documents = [
            "Quantum computing uses quantum mechanics for computation",
            "Classical computers use binary logic", 
            "Weather forecast for tomorrow"
        ]
        
        results = qpmel_reranker.rerank(query, documents, method="quantum")
        print(f"QPMeL reranking results: {len(results)} documents")
        
    except FileNotFoundError:
        print("QPMeL model not found - train one first with train_qpmel.py")
    
    # Example 2: Upgrading existing reranker
    print("\nExample 2: Upgrading existing reranker")
    original_reranker = QuantumRAGReranker()
    
    try:
        enhanced_reranker = upgrade_existing_reranker(
            original_reranker, 
            "models/qpmel_trained.pt"
        )
        print("Successfully upgraded reranker with QPMeL")
    except FileNotFoundError:
        print("QPMeL model not found for upgrade")
    
    # Example 3: Integration info
    print("\nExample 3: Model information")
    try:
        engine = create_qpmel_similarity_engine("models/qpmel_trained.pt")
        info = engine.get_model_info()
        print(f"QPMeL integration status: {info.get('qpmel_integrated', False)}")
    except FileNotFoundError:
        print("QPMeL model not found for info display")

if __name__ == "__main__":
    demonstrate_qpmel_integration()