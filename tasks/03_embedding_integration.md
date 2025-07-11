# Task 03: SentenceTransformer Integration and Embedding Processing

## Objective
Integrate SentenceTransformers with quantum circuits, implementing the embedding preprocessing pipeline as specified in the PRD and documentation.

## Prerequisites
- Task 01: Environment Setup completed
- Task 02: Basic Quantum Circuits implemented
- SentenceTransformers library installed and verified

## Technical Reference
- **PRD Section 2.2**: Implementation Stack - SentenceTransformers
- **PRD Section 4.1**: System Requirements - Embedding Models
- **PRD Section 5.2**: Integration with Existing RAG Pipeline
- **Documentation**: "Recommendation for Pre-trained Text Embedding Mode.md"
- **Documentation**: "Quantum-Inspired Semantic Reranking with PyTorch_.md"

## Implementation Steps

### 1. Create Embedding Handler Module
```python
# quantum_rerank/core/embeddings.py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding processing."""
    model_name: str = 'all-mpnet-base-v2'  # From docs recommendation
    embedding_dim: int = 768
    max_sequence_length: int = 512
    batch_size: int = 32
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    normalize_embeddings: bool = True

class EmbeddingProcessor:
    """
    Handles text embedding generation and preprocessing for quantum circuits.
    
    Based on PRD Section 5.2 and documentation recommendations.
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        
        # Determine device
        if self.config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.config.device
        
        # Load SentenceTransformer model
        try:
            self.model = SentenceTransformer(self.config.model_name, device=self.device)
            logger.info(f"Loaded {self.config.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Verify embedding dimension matches configuration
        test_embedding = self.model.encode(["test"], convert_to_tensor=False)
        actual_dim = len(test_embedding[0])
        
        if actual_dim != self.config.embedding_dim:
            logger.warning(f"Model dimension {actual_dim} != config {self.config.embedding_dim}")
            self.config.embedding_dim = actual_dim
    
    def encode_texts(self, texts: List[str], 
                    batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode list of texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Override default batch size
            
        Returns:
            Array of embeddings [n_texts, embedding_dim]
        """
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.config.batch_size
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=len(texts) > 100
            )
            
            logger.debug(f"Encoded {len(texts)} texts to shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Encode single text to embedding.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector [embedding_dim]
        """
        embedding = self.encode_texts([text])
        return embedding[0] if len(embedding) > 0 else np.array([])
```

### 2. Implement Quantum-Compatible Preprocessing
```python
def preprocess_for_quantum(self, embeddings: np.ndarray, 
                          n_qubits: int = 4) -> Tuple[np.ndarray, dict]:
    """
    Preprocess embeddings for quantum circuit encoding.
    
    Based on PRD quantum constraints and circuit requirements.
    
    Args:
        embeddings: Input embeddings [n_embeddings, embedding_dim]
        n_qubits: Number of qubits for quantum encoding
        
    Returns:
        Tuple of (processed_embeddings, metadata)
    """
    max_amplitudes = 2 ** n_qubits  # Maximum amplitudes for quantum state
    
    processed_embeddings = []
    metadata = {
        'original_dim': embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings),
        'target_amplitudes': max_amplitudes,
        'n_qubits': n_qubits,
        'processing_applied': []
    }
    
    for embedding in embeddings:
        processed_emb = embedding.copy()
        
        # Step 1: Dimensionality adjustment
        if len(processed_emb) > max_amplitudes:
            # Truncate to fit quantum state
            processed_emb = processed_emb[:max_amplitudes]
            metadata['processing_applied'].append('truncation')
        elif len(processed_emb) < max_amplitudes:
            # Pad with zeros
            padding = max_amplitudes - len(processed_emb)
            processed_emb = np.pad(processed_emb, (0, padding), mode='constant')
            metadata['processing_applied'].append('zero_padding')
        
        # Step 2: Ensure unit norm (required for quantum states)
        norm = np.linalg.norm(processed_emb)
        if norm > 0:
            processed_emb = processed_emb / norm
            metadata['processing_applied'].append('normalization')
        
        processed_embeddings.append(processed_emb)
    
    result = np.array(processed_embeddings)
    
    logger.debug(f"Quantum preprocessing: {metadata}")
    return result, metadata

def create_embedding_batches(self, texts: List[str], 
                           batch_size: Optional[int] = None) -> List[Tuple[List[str], np.ndarray]]:
    """
    Create batches of texts and their embeddings for efficient processing.
    
    Supports PRD batch processing requirements (50-100 documents).
    """
    batch_size = batch_size or self.config.batch_size
    batches = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = self.encode_texts(batch_texts)
        batches.append((batch_texts, batch_embeddings))
        
        logger.debug(f"Created batch {len(batches)}: {len(batch_texts)} texts")
    
    return batches

def compute_classical_similarity(self, embedding1: np.ndarray, 
                               embedding2: np.ndarray) -> float:
    """
    Compute classical cosine similarity for comparison baseline.
    
    Args:
        embedding1, embedding2: Normalized embedding vectors
        
    Returns:
        Cosine similarity score [0, 1]
    """
    # Ensure embeddings are normalized
    emb1_norm = embedding1 / np.linalg.norm(embedding1)
    emb2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Compute cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    
    # Clamp to [0, 1] range
    similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
    
    return float(similarity)
```

### 3. Add Integration with Quantum Circuits
```python
# quantum_rerank/core/quantum_embedding_bridge.py
from .embeddings import EmbeddingProcessor
from .quantum_circuits import BasicQuantumCircuits
import numpy as np
from typing import List, Tuple, Dict

class QuantumEmbeddingBridge:
    """
    Bridge between classical embeddings and quantum circuits.
    
    Implements the integration specified in PRD Section 5.2.
    """
    
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.embedding_processor = EmbeddingProcessor()
        self.quantum_circuits = BasicQuantumCircuits(n_qubits=n_qubits)
        
    def text_to_quantum_circuit(self, text: str) -> Tuple[QuantumCircuit, dict]:
        """
        Convert text directly to quantum circuit.
        
        Full pipeline: text -> embedding -> quantum circuit
        """
        # Step 1: Generate embedding
        embedding = self.embedding_processor.encode_single_text(text)
        
        # Step 2: Preprocess for quantum
        processed_embeddings, metadata = self.embedding_processor.preprocess_for_quantum(
            np.array([embedding]), self.n_qubits
        )
        processed_embedding = processed_embeddings[0]
        
        # Step 3: Create quantum circuit
        quantum_circuit = self.quantum_circuits.amplitude_encode_embedding(processed_embedding)
        
        # Add metadata
        metadata.update({
            'text_length': len(text),
            'original_embedding_dim': len(embedding),
            'quantum_circuit_depth': quantum_circuit.depth(),
            'quantum_circuit_size': quantum_circuit.size()
        })
        
        return quantum_circuit, metadata
    
    def batch_texts_to_circuits(self, texts: List[str]) -> List[Tuple[str, QuantumCircuit, dict]]:
        """
        Convert batch of texts to quantum circuits efficiently.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (text, circuit, metadata) tuples
        """
        # Batch encode all texts
        embeddings = self.embedding_processor.encode_texts(texts)
        
        # Preprocess for quantum
        processed_embeddings, batch_metadata = self.embedding_processor.preprocess_for_quantum(
            embeddings, self.n_qubits
        )
        
        results = []
        for i, (text, processed_embedding) in enumerate(zip(texts, processed_embeddings)):
            # Create quantum circuit
            circuit = self.quantum_circuits.amplitude_encode_embedding(processed_embedding)
            
            # Individual metadata
            metadata = {
                'batch_index': i,
                'text_length': len(text),
                'quantum_circuit_depth': circuit.depth(),
                'quantum_circuit_size': circuit.size(),
                'batch_metadata': batch_metadata
            }
            
            results.append((text, circuit, metadata))
        
        return results
```

### 4. Performance Monitoring and Benchmarking
```python
def benchmark_embedding_performance(self) -> Dict:
    """
    Benchmark embedding performance against PRD targets.
    
    Returns metrics for similarity computation speed, batch processing, etc.
    """
    import time
    
    # Test data
    test_texts = [
        "Quantum computing uses quantum mechanics for computation",
        "Machine learning algorithms process data to find patterns",
        "Information retrieval systems find relevant documents",
        "Natural language processing analyzes human language"
    ]
    
    results = {}
    
    # Single text encoding
    start_time = time.time()
    single_embedding = self.encode_single_text(test_texts[0])
    single_time = time.time() - start_time
    
    results['single_encoding_ms'] = single_time * 1000
    
    # Batch encoding
    start_time = time.time()
    batch_embeddings = self.encode_texts(test_texts)
    batch_time = time.time() - start_time
    
    results['batch_encoding_ms'] = batch_time * 1000
    results['batch_per_text_ms'] = (batch_time / len(test_texts)) * 1000
    
    # Quantum preprocessing
    start_time = time.time()
    processed, metadata = self.preprocess_for_quantum(batch_embeddings, n_qubits=4)
    preprocessing_time = time.time() - start_time
    
    results['quantum_preprocessing_ms'] = preprocessing_time * 1000
    
    # Classical similarity (baseline)
    start_time = time.time()
    similarity = self.compute_classical_similarity(batch_embeddings[0], batch_embeddings[1])
    similarity_time = time.time() - start_time
    
    results['classical_similarity_ms'] = similarity_time * 1000
    
    # Memory usage estimation
    import sys
    results['embedding_memory_mb'] = sys.getsizeof(batch_embeddings) / (1024 * 1024)
    
    return results

def validate_embedding_quality(self) -> Dict:
    """
    Validate embedding quality and quantum compatibility.
    """
    test_texts = [
        "quantum computing",
        "classical computing", 
        "machine learning",
        "artificial intelligence"
    ]
    
    embeddings = self.encode_texts(test_texts)
    
    # Check embedding properties
    results = {
        'embedding_dim': embeddings.shape[1],
        'all_finite': np.all(np.isfinite(embeddings)),
        'normalized': np.allclose(np.linalg.norm(embeddings, axis=1), 1.0),
        'embedding_range': {
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings))
        }
    }
    
    # Test quantum preprocessing
    processed, metadata = self.preprocess_for_quantum(embeddings, n_qubits=4)
    results['quantum_compatible'] = all([
        np.allclose(np.linalg.norm(emb), 1.0) for emb in processed
    ])
    
    # Test similarity relationships
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = self.compute_classical_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    results['similarity_stats'] = {
        'mean': float(np.mean(similarities)),
        'min': float(np.min(similarities)),
        'max': float(np.max(similarities)),
        'std': float(np.std(similarities))
    }
    
    return results
```

## Success Criteria

### Functional Requirements
- [ ] SentenceTransformer model loads correctly with recommended model
- [ ] Text encoding produces embeddings of expected dimension (768)
- [ ] Quantum preprocessing correctly adjusts embeddings for 2^n amplitudes
- [ ] Integration bridge converts texts to quantum circuits successfully
- [ ] Batch processing handles 50-100 documents efficiently

### Performance Requirements
- [ ] Single text encoding <100ms (supporting PRD similarity targets)
- [ ] Batch encoding scales linearly with text count
- [ ] Quantum preprocessing <10ms per embedding
- [ ] Memory usage stays within PRD bounds (<2GB for 100 docs)

### Quality Requirements
- [ ] Embeddings are properly normalized
- [ ] Classical similarity computation works correctly
- [ ] Quantum-preprocessed embeddings maintain semantic relationships
- [ ] No loss of critical information during dimension adjustment

## Files to Create
```
quantum_rerank/core/
├── embeddings.py
├── quantum_embedding_bridge.py
└── embedding_validators.py

tests/unit/
├── test_embeddings.py
├── test_quantum_bridge.py
└── test_embedding_quality.py

examples/
├── embedding_demo.py
└── quantum_embedding_demo.py

benchmarks/
└── embedding_performance.py
```

## Testing & Validation

### Unit Tests
```python
def test_embedding_processor():
    processor = EmbeddingProcessor()
    embeddings = processor.encode_texts(["test text"])
    assert embeddings.shape[1] == processor.config.embedding_dim

def test_quantum_preprocessing():
    processor = EmbeddingProcessor()
    embeddings = processor.encode_texts(["test"])
    processed, metadata = processor.preprocess_for_quantum(embeddings, n_qubits=4)
    assert processed.shape[1] == 16  # 2^4
    assert np.allclose(np.linalg.norm(processed[0]), 1.0)

def test_quantum_bridge():
    bridge = QuantumEmbeddingBridge(n_qubits=4)
    circuit, metadata = bridge.text_to_quantum_circuit("test text")
    assert circuit.num_qubits == 4
    assert bridge.quantum_circuits.validate_circuit_constraints(circuit)
```

### Integration Tests
```python
def test_full_pipeline():
    bridge = QuantumEmbeddingBridge(n_qubits=4)
    
    # Test single text
    circuit, metadata = bridge.text_to_quantum_circuit("quantum computing")
    assert 'quantum_circuit_depth' in metadata
    
    # Test batch processing
    texts = ["quantum", "classical", "hybrid"]
    results = bridge.batch_texts_to_circuits(texts)
    assert len(results) == 3
    
    for text, circuit, metadata in results:
        assert circuit.num_qubits == 4
        assert 'quantum_circuit_depth' in metadata
```

### Performance Tests
```python
def test_performance_benchmarks():
    processor = EmbeddingProcessor()
    results = processor.benchmark_embedding_performance()
    
    # Check PRD compliance
    assert results['single_encoding_ms'] < 1000  # Reasonable for development
    assert results['classical_similarity_ms'] < 10  # Should be very fast
    assert results['quantum_preprocessing_ms'] < 100  # Should be efficient
```

## Next Task Dependencies
This task enables:
- Task 04: SWAP Test Implementation (needs quantum circuits from embeddings)
- Task 05: Quantum Parameter Prediction (needs embedding preprocessing)
- Task 06: Quantum Similarity Engine (needs embedding-circuit bridge)

## References
- PRD Section 2.2: Implementation Stack
- PRD Section 5.2: Integration with RAG Pipeline
- Documentation: Embedding model recommendations
- Documentation: PyTorch integration guide