# Task 05: Quantum Parameter Prediction with Classical MLP

## Objective
Implement a classical MLP that predicts quantum circuit parameters from embeddings, enabling the hybrid quantum-classical training pipeline as specified in the PRD.

## Prerequisites
- Task 01: Environment Setup completed
- Task 02: Basic Quantum Circuits implemented
- Task 03: Embedding Integration completed
- Task 04: SWAP Test Implementation completed
- PyTorch installed and verified

## Technical Reference
- **PRD Section 3.1**: Core Algorithms - Parameterized Quantum Circuits (PQC)
- **PRD Section 2.2**: Implementation Stack - Classical ML (PyTorch)
- **PRD Section 5.1**: Quantum-Inspired Similarity Engine
- **Documentation**: "Quantum-Inspired Semantic Reranking with PyTorch_.md"
- **Documentation**: "Comprehensive PennyLane Documentation for Quantum-.md"
- **Research Papers**: Quantum parameter prediction and hybrid training

## Implementation Steps

### 1. Create Parameter Prediction Module
```python
# quantum_rerank/ml/parameter_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParameterPredictorConfig:
    """Configuration for quantum parameter prediction."""
    embedding_dim: int = 768  # From SentenceTransformers
    hidden_dims: List[int] = None  # Will default to [512, 256]
    n_qubits: int = 4
    n_layers: int = 2  # Quantum circuit layers
    dropout_rate: float = 0.1
    activation: str = 'relu'  # 'relu', 'tanh', 'gelu'
    parameter_range: str = 'pi'  # 'pi' for [0, π], '2pi' for [0, 2π]
    device: str = 'auto'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]

class QuantumParameterPredictor(nn.Module):
    """
    Classical MLP that predicts quantum circuit parameters from embeddings.
    
    Based on PRD Section 3.1 and hybrid quantum-classical approach.
    """
    
    def __init__(self, config: ParameterPredictorConfig = None):
        super().__init__()
        
        self.config = config or ParameterPredictorConfig()
        
        # Calculate number of parameters needed
        self.params_per_layer = self._calculate_params_per_layer()
        self.total_params = self.params_per_layer * self.config.n_layers
        
        # Build MLP layers
        self.layers = self._build_mlp_layers()
        
        # Parameter output heads
        self.parameter_heads = self._build_parameter_heads()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Parameter predictor initialized: {self.total_params} quantum parameters")
    
    def _calculate_params_per_layer(self) -> int:
        """
        Calculate number of parameters needed per quantum circuit layer.
        
        Based on standard parameterized quantum circuit structure:
        - 3 rotation parameters per qubit (RY, RZ, RY)
        - Entangling parameters between adjacent qubits
        """
        # Rotation parameters: 3 per qubit
        rotation_params = 3 * self.config.n_qubits
        
        # Entangling parameters: between adjacent qubits
        entangling_params = self.config.n_qubits - 1
        
        return rotation_params + entangling_params
    
    def _build_mlp_layers(self) -> nn.ModuleList:
        """Build the main MLP layers."""
        layers = nn.ModuleList()
        
        # Input layer
        prev_dim = self.config.embedding_dim
        
        # Hidden layers
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return layers
    
    def _build_parameter_heads(self) -> nn.ModuleDict:
        """Build separate heads for different parameter types."""
        heads = nn.ModuleDict()
        
        final_hidden_dim = self.config.hidden_dims[-1]
        
        # Rotation parameter heads (one for each type)
        heads['ry_params'] = nn.Linear(final_hidden_dim, 
                                      self.config.n_qubits * self.config.n_layers)
        heads['rz_params'] = nn.Linear(final_hidden_dim, 
                                      self.config.n_qubits * self.config.n_layers)
        heads['ry2_params'] = nn.Linear(final_hidden_dim, 
                                       self.config.n_qubits * self.config.n_layers)
        
        # Entangling parameter head
        heads['entangling_params'] = nn.Linear(final_hidden_dim, 
                                              (self.config.n_qubits - 1) * self.config.n_layers)
        
        return heads
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'tanh':
            return nn.Tanh()
        elif self.config.activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()  # Default
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: embeddings -> quantum parameters.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Dictionary of parameter tensors
        """
        # Forward through main MLP
        x = embeddings
        for layer in self.layers:
            x = layer(x)
        
        # Generate parameters through separate heads
        parameters = {}
        
        # Rotation parameters with appropriate scaling
        parameters['ry_params'] = self._scale_parameters(
            torch.sigmoid(self.parameter_heads['ry_params'](x))
        )
        parameters['rz_params'] = self._scale_parameters(
            torch.sigmoid(self.parameter_heads['rz_params'](x))
        )
        parameters['ry2_params'] = self._scale_parameters(
            torch.sigmoid(self.parameter_heads['ry2_params'](x))
        )
        
        # Entangling parameters
        parameters['entangling_params'] = self._scale_parameters(
            torch.sigmoid(self.parameter_heads['entangling_params'](x))
        )
        
        return parameters
    
    def _scale_parameters(self, sigmoid_output: torch.Tensor) -> torch.Tensor:
        """Scale sigmoid output to appropriate parameter range."""
        if self.config.parameter_range == 'pi':
            return sigmoid_output * torch.pi
        elif self.config.parameter_range == '2pi':
            return sigmoid_output * 2 * torch.pi
        else:
            return sigmoid_output * torch.pi  # Default
    
    def get_flat_parameters(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get all parameters as a flat tensor for compatibility.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Flat parameter tensor [batch_size, total_params]
        """
        param_dict = self.forward(embeddings)
        
        # Concatenate all parameter types
        flat_params = torch.cat([
            param_dict['ry_params'],
            param_dict['rz_params'], 
            param_dict['ry2_params'],
            param_dict['entangling_params']
        ], dim=1)
        
        return flat_params
```

### 2. Implement Quantum Circuit Parameter Integration
```python
# quantum_rerank/ml/parameterized_circuits.py
from ..core.quantum_circuits import BasicQuantumCircuits
from .parameter_predictor import QuantumParameterPredictor
import torch
import numpy as np
from qiskit import QuantumCircuit
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class ParameterizedQuantumCircuits:
    """
    Creates parameterized quantum circuits using predicted parameters.
    
    Bridges classical parameter prediction with quantum circuit construction.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.quantum_circuits = BasicQuantumCircuits(n_qubits)
        
    def create_parameterized_circuit(self, 
                                   parameters: Dict[str, torch.Tensor],
                                   batch_index: int = 0) -> QuantumCircuit:
        """
        Create a parameterized quantum circuit from predicted parameters.
        
        Args:
            parameters: Dictionary of parameter tensors from predictor
            batch_index: Which sample in the batch to use
            
        Returns:
            Parameterized quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits, name="parameterized_circuit")
        
        # Extract parameters for this sample
        ry_params = parameters['ry_params'][batch_index].detach().cpu().numpy()
        rz_params = parameters['rz_params'][batch_index].detach().cpu().numpy()
        ry2_params = parameters['ry2_params'][batch_index].detach().cpu().numpy()
        entangling_params = parameters['entangling_params'][batch_index].detach().cpu().numpy()
        
        # Build circuit layer by layer
        for layer in range(self.n_layers):
            # Rotation gates for each qubit
            for qubit in range(self.n_qubits):
                param_idx = layer * self.n_qubits + qubit
                
                qc.ry(ry_params[param_idx], qubit)
                qc.rz(rz_params[param_idx], qubit)
                qc.ry(ry2_params[param_idx], qubit)
            
            # Entangling gates between adjacent qubits
            for qubit in range(self.n_qubits - 1):
                param_idx = layer * (self.n_qubits - 1) + qubit
                qc.rzz(entangling_params[param_idx], qubit, qubit + 1)
        
        return qc
    
    def create_batch_circuits(self, 
                            parameters: Dict[str, torch.Tensor]) -> List[QuantumCircuit]:
        """
        Create multiple parameterized circuits from a batch of parameters.
        
        Args:
            parameters: Batch of parameter tensors
            
        Returns:
            List of parameterized quantum circuits
        """
        batch_size = parameters['ry_params'].shape[0]
        circuits = []
        
        for i in range(batch_size):
            circuit = self.create_parameterized_circuit(parameters, i)
            circuits.append(circuit)
        
        return circuits
    
    def validate_circuit_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict:
        """
        Validate that predicted parameters are within expected ranges.
        
        Returns validation results and statistics.
        """
        validation_results = {}
        
        for param_type, param_tensor in parameters.items():
            param_numpy = param_tensor.detach().cpu().numpy()
            
            validation_results[param_type] = {
                'shape': param_tensor.shape,
                'min': float(np.min(param_numpy)),
                'max': float(np.max(param_numpy)),
                'mean': float(np.mean(param_numpy)),
                'std': float(np.std(param_numpy)),
                'finite': bool(np.all(np.isfinite(param_numpy))),
                'in_range': bool(np.all((param_numpy >= 0) & (param_numpy <= 2 * np.pi)))
            }
        
        # Overall validation
        all_finite = all(result['finite'] for result in validation_results.values())
        all_in_range = all(result['in_range'] for result in validation_results.values())
        
        validation_results['overall'] = {
            'all_finite': all_finite,
            'all_in_range': all_in_range,
            'valid': all_finite and all_in_range
        }
        
        return validation_results
```

### 3. Implement Training Pipeline
```python
# quantum_rerank/ml/training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for parameter predictor training."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    validation_split: float = 0.2
    patience: int = 10  # Early stopping
    min_delta: float = 1e-4  # Minimum improvement
    weight_decay: float = 1e-5
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1

class FidelityTripletLoss(nn.Module):
    """
    Triplet loss using quantum fidelity as similarity metric.
    
    Based on research papers and PRD hybrid training approach.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, 
                anchor_fidelity: torch.Tensor,
                positive_fidelity: torch.Tensor,
                negative_fidelity: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss using fidelity values.
        
        Args:
            anchor_fidelity: Fidelity between anchor and positive
            positive_fidelity: Should be same as anchor_fidelity
            negative_fidelity: Fidelity between anchor and negative
            
        Returns:
            Triplet loss value
        """
        # Triplet loss: maximize positive fidelity, minimize negative fidelity
        loss = torch.clamp(
            self.margin - anchor_fidelity + negative_fidelity, 
            min=0.0
        )
        
        return loss.mean()

class ParameterPredictorTrainer:
    """
    Trainer for quantum parameter predictor using fidelity-based loss.
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
        
        # Loss function
        self.triplet_loss = FidelityTripletLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def create_triplet_dataset(self, 
                             embeddings: np.ndarray,
                             similarity_labels: Optional[np.ndarray] = None) -> TensorDataset:
        """
        Create triplet dataset from embeddings.
        
        Args:
            embeddings: Array of embeddings [n_samples, embedding_dim]
            similarity_labels: Optional similarity labels for supervised triplets
            
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
                # Use labels if available
                positive_candidates = np.where(similarity_labels[i] > 0.7)[0]
                if len(positive_candidates) > 0:
                    pos_idx = np.random.choice(positive_candidates)
                else:
                    pos_idx = np.random.choice(n_samples)  # Fallback
            else:
                # Random positive (could be improved with actual similarity)
                pos_idx = np.random.choice(n_samples)
            
            positive = embeddings[pos_idx]
            
            # Find negative (dissimilar sample)
            if similarity_labels is not None:
                negative_candidates = np.where(similarity_labels[i] < 0.3)[0]
                if len(negative_candidates) > 0:
                    neg_idx = np.random.choice(negative_candidates)
                else:
                    neg_idx = np.random.choice(n_samples)  # Fallback
            else:
                # Random negative
                neg_idx = np.random.choice(n_samples)
            
            negative = embeddings[neg_idx]
            
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
        
        # Convert to tensors
        anchor_tensor = torch.FloatTensor(np.array(anchors))
        positive_tensor = torch.FloatTensor(np.array(positives))
        negative_tensor = torch.FloatTensor(np.array(negatives))
        
        return TensorDataset(anchor_tensor, positive_tensor, negative_tensor)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (anchors, positives, negatives) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Predict parameters for all triplet components
            anchor_params = self.model(anchors)
            positive_params = self.model(positives)
            negative_params = self.model(negatives)
            
            # For training, we'll use a simplified loss based on parameter similarity
            # In practice, this would involve quantum circuit simulation
            anchor_flat = self.model.get_flat_parameters(anchors)
            positive_flat = self.model.get_flat_parameters(positives)
            negative_flat = self.model.get_flat_parameters(negatives)
            
            # Compute parameter-based similarities (proxy for fidelity)
            anchor_pos_sim = F.cosine_similarity(anchor_flat, positive_flat, dim=1)
            anchor_neg_sim = F.cosine_similarity(anchor_flat, negative_flat, dim=1)
            
            # Triplet loss
            loss = self.triplet_loss(anchor_pos_sim, anchor_pos_sim, anchor_neg_sim)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for anchors, positives, negatives in val_loader:
                anchor_flat = self.model.get_flat_parameters(anchors)
                positive_flat = self.model.get_flat_parameters(positives)
                negative_flat = self.model.get_flat_parameters(negatives)
                
                anchor_pos_sim = F.cosine_similarity(anchor_flat, positive_flat, dim=1)
                anchor_neg_sim = F.cosine_similarity(anchor_flat, negative_flat, dim=1)
                
                loss = self.triplet_loss(anchor_pos_sim, anchor_pos_sim, anchor_neg_sim)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, dataset: TensorDataset) -> Dict:
        """
        Train the parameter predictor.
        
        Args:
            dataset: Triplet dataset for training
            
        Returns:
            Training history and final metrics
        """
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
        
        logger.info(f"Starting training: {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
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
        
        training_results = {
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.history['train_loss']),
            'history': self.history
        }
        
        logger.info("Training completed")
        return training_results
```

### 4. Add Integration and Testing
```python
# quantum_rerank/ml/parameter_integration.py
from .parameter_predictor import QuantumParameterPredictor, ParameterPredictorConfig
from .parameterized_circuits import ParameterizedQuantumCircuits
from ..core.embeddings import EmbeddingProcessor
import torch
import numpy as np
from typing import List, Tuple, Dict

class EmbeddingToCircuitPipeline:
    """
    Complete pipeline: embedding -> parameters -> quantum circuit.
    
    Integrates all components for end-to-end processing.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize components
        self.embedding_processor = EmbeddingProcessor()
        
        config = ParameterPredictorConfig(
            embedding_dim=self.embedding_processor.config.embedding_dim,
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        self.parameter_predictor = QuantumParameterPredictor(config)
        self.circuit_builder = ParameterizedQuantumCircuits(n_qubits, n_layers)
    
    def text_to_parameterized_circuit(self, text: str) -> Tuple[QuantumCircuit, Dict]:
        """
        Convert text to parameterized quantum circuit.
        
        Complete pipeline: text -> embedding -> parameters -> circuit
        """
        # Step 1: Generate embedding
        embedding = self.embedding_processor.encode_single_text(text)
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0)
        
        # Step 2: Predict parameters
        with torch.no_grad():
            parameters = self.parameter_predictor(embedding_tensor)
        
        # Step 3: Create circuit
        circuit = self.circuit_builder.create_parameterized_circuit(parameters, 0)
        
        # Collect metadata
        metadata = {
            'text_length': len(text),
            'embedding_dim': len(embedding),
            'total_parameters': sum(p.numel() for p in parameters.values()),
            'circuit_depth': circuit.depth(),
            'circuit_size': circuit.size()
        }
        
        return circuit, metadata
    
    def batch_text_to_circuits(self, texts: List[str]) -> List[Tuple[str, QuantumCircuit, Dict]]:
        """
        Convert batch of texts to parameterized circuits.
        
        Efficient batch processing version.
        """
        # Batch encode embeddings
        embeddings = self.embedding_processor.encode_texts(texts)
        embedding_tensor = torch.FloatTensor(embeddings)
        
        # Batch predict parameters
        with torch.no_grad():
            parameters = self.parameter_predictor(embedding_tensor)
        
        # Create circuits
        circuits = self.circuit_builder.create_batch_circuits(parameters)
        
        # Combine results
        results = []
        for i, (text, circuit) in enumerate(zip(texts, circuits)):
            metadata = {
                'batch_index': i,
                'text_length': len(text),
                'circuit_depth': circuit.depth(),
                'circuit_size': circuit.size()
            }
            results.append((text, circuit, metadata))
        
        return results
    
    def benchmark_pipeline_performance(self) -> Dict:
        """Benchmark the complete pipeline performance."""
        import time
        
        test_texts = [
            "quantum computing applications",
            "machine learning algorithms", 
            "information retrieval systems",
            "natural language processing"
        ]
        
        results = {}
        
        # Single text processing
        start_time = time.time()
        circuit, metadata = self.text_to_parameterized_circuit(test_texts[0])
        single_time = time.time() - start_time
        
        results['single_processing_ms'] = single_time * 1000
        
        # Batch processing
        start_time = time.time()
        batch_results = self.batch_text_to_circuits(test_texts)
        batch_time = time.time() - start_time
        
        results['batch_processing_ms'] = batch_time * 1000
        results['batch_per_item_ms'] = (batch_time / len(test_texts)) * 1000
        
        # Parameter statistics
        with torch.no_grad():
            embeddings = self.embedding_processor.encode_texts(test_texts)
            embedding_tensor = torch.FloatTensor(embeddings)
            parameters = self.parameter_predictor(embedding_tensor)
            
            param_validation = self.circuit_builder.validate_circuit_parameters(parameters)
            results['parameter_validation'] = param_validation
        
        return results
```

## Success Criteria

### Functional Requirements
- [ ] Parameter predictor accepts 768-d embeddings and outputs valid quantum parameters
- [ ] Generated parameters are in correct ranges [0, π] or [0, 2π]
- [ ] Parameterized circuits can be created from predicted parameters
- [ ] Circuit depth stays ≤15 gates (PRD constraint)
- [ ] Batch processing works efficiently for multiple texts
- [ ] Training pipeline can optimize parameters using triplet loss

### Performance Requirements
- [ ] Parameter prediction <50ms per embedding
- [ ] Circuit creation <10ms per parameter set
- [ ] Memory usage scales reasonably with batch size
- [ ] Training converges within reasonable epochs

### Quality Requirements
- [ ] Parameters produce valid quantum circuits
- [ ] Parameter validation passes all checks
- [ ] Integration pipeline works end-to-end
- [ ] Training loss decreases over epochs

## Files to Create
```
quantum_rerank/ml/
├── __init__.py
├── parameter_predictor.py
├── parameterized_circuits.py
├── training.py
└── parameter_integration.py

tests/unit/
├── test_parameter_predictor.py
├── test_parameterized_circuits.py
├── test_training.py
└── test_parameter_integration.py

examples/
├── parameter_prediction_demo.py
└── training_demo.py

benchmarks/
└── parameter_prediction_performance.py
```

## Testing & Validation

### Unit Tests
```python
def test_parameter_predictor():
    config = ParameterPredictorConfig(embedding_dim=768, n_qubits=4)
    model = QuantumParameterPredictor(config)
    
    embeddings = torch.randn(2, 768)
    parameters = model(embeddings)
    
    assert 'ry_params' in parameters
    assert parameters['ry_params'].shape == (2, 8)  # 4 qubits * 2 layers

def test_parameterized_circuits():
    circuits = ParameterizedQuantumCircuits(n_qubits=4, n_layers=2)
    
    # Mock parameters
    parameters = {
        'ry_params': torch.randn(1, 8),
        'rz_params': torch.randn(1, 8),
        'ry2_params': torch.randn(1, 8),
        'entangling_params': torch.randn(1, 6)
    }
    
    circuit = circuits.create_parameterized_circuit(parameters, 0)
    assert circuit.num_qubits == 4
    assert circuit.depth() <= 15  # PRD constraint

def test_pipeline_integration():
    pipeline = EmbeddingToCircuitPipeline(n_qubits=4)
    
    circuit, metadata = pipeline.text_to_parameterized_circuit("test text")
    assert circuit.num_qubits == 4
    assert 'circuit_depth' in metadata
```

### Integration Tests
```python
def test_end_to_end_pipeline():
    pipeline = EmbeddingToCircuitPipeline(n_qubits=4)
    
    texts = ["quantum computing", "machine learning"]
    results = pipeline.batch_text_to_circuits(texts)
    
    assert len(results) == 2
    for text, circuit, metadata in results:
        assert circuit.num_qubits == 4
        assert circuit.depth() <= 15

def test_training_pipeline():
    config = ParameterPredictorConfig(embedding_dim=768, n_qubits=4)
    model = QuantumParameterPredictor(config)
    trainer = ParameterPredictorTrainer(model)
    
    # Create dummy dataset
    embeddings = np.random.randn(100, 768)
    dataset = trainer.create_triplet_dataset(embeddings)
    
    # Quick training test
    config.num_epochs = 2
    results = trainer.train(dataset)
    assert 'final_train_loss' in results
```

### Performance Tests
```python
def test_performance_benchmarks():
    pipeline = EmbeddingToCircuitPipeline(n_qubits=4)
    results = pipeline.benchmark_pipeline_performance()
    
    # Check performance targets
    assert results['single_processing_ms'] < 1000  # Reasonable for development
    assert results['parameter_validation']['overall']['valid']
```

## Next Task Dependencies
This task enables:
- Task 06: Basic Quantum Similarity Engine (parameterized circuits ready)
- Task 11: Hybrid Quantum-Classical Training (training pipeline ready)
- Task 12: Batch Processing Optimization (parameter prediction scaling)

## References
- PRD Section 3.1: Core Algorithms - PQC
- PRD Section 5.1: Quantum-Inspired Similarity Engine
- Documentation: PyTorch integration guide
- Documentation: PennyLane hybrid training
- Research Papers: Quantum parameter prediction