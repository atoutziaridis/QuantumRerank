# Task 02: Basic Quantum Circuit Creation and Simulation

## Objective
Implement basic quantum circuit creation and simulation using Qiskit, focusing on the small-scale circuits (2-4 qubits, ≤15 gates) specified in the PRD.

## Prerequisites
- Task 01: Environment Setup completed
- Qiskit and Qiskit-Aer installed and verified
- Basic Python project structure in place

## Technical Reference
- **PRD Section 1.3**: Proven Implementation Patterns
- **PRD Section 4.1**: System Requirements (2-4 qubits, ≤15 gates)
- **PRD Section 3.1**: Core Algorithms - Amplitude Encoding
- **Documentation**: "Provide detailed documentation for using Qiskit to.md"
- **Research**: Quantum circuit examples from papers

## Implementation Steps

### 1. Create Basic Quantum Circuit Module
```python
# quantum_rerank/core/quantum_circuits.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BasicQuantumCircuits:
    """
    Basic quantum circuit operations for QuantumRerank.
    Implements small-scale circuits as specified in PRD Section 4.1.
    """
    
    def __init__(self, n_qubits: int = 4, max_depth: int = 15):
        """
        Initialize quantum circuit handler.
        
        Args:
            n_qubits: Number of qubits (2-4 as per PRD)
            max_depth: Maximum circuit depth (≤15 as per PRD)
        """
        if not 2 <= n_qubits <= 4:
            raise ValueError("n_qubits must be between 2 and 4 (PRD requirement)")
        
        self.n_qubits = n_qubits
        self.max_depth = max_depth
        self.simulator = AerSimulator(method='statevector')
        
        logger.info(f"Initialized BasicQuantumCircuits with {n_qubits} qubits")
    
    def create_empty_circuit(self) -> QuantumCircuit:
        """Create an empty quantum circuit with specified qubits."""
        qc = QuantumCircuit(self.n_qubits, name="empty_circuit")
        return qc
    
    def create_superposition_circuit(self) -> QuantumCircuit:
        """Create a simple superposition circuit for testing."""
        qc = QuantumCircuit(self.n_qubits, name="superposition")
        
        # Apply Hadamard to all qubits
        for qubit in range(self.n_qubits):
            qc.h(qubit)
        
        return qc
    
    def create_entanglement_circuit(self) -> QuantumCircuit:
        """Create a basic entanglement circuit."""
        qc = QuantumCircuit(self.n_qubits, name="entanglement")
        
        # Create entanglement between adjacent qubits
        qc.h(0)  # Superposition on first qubit
        for i in range(self.n_qubits - 1):
            qc.cnot(i, i + 1)  # Entangle adjacent qubits
        
        return qc
```

### 2. Implement Amplitude Encoding
Based on PRD Section 3.1 and research papers:
```python
def amplitude_encode_embedding(self, embedding: np.ndarray) -> QuantumCircuit:
    """
    Encode classical embedding into quantum state via amplitude encoding.
    
    Implementation based on PRD Section 3.1 and quantum papers.
    
    Args:
        embedding: Classical embedding vector
        
    Returns:
        Quantum circuit with amplitude-encoded state
    """
    # Normalize and pad embedding to fit circuit size
    max_amplitudes = 2 ** self.n_qubits
    
    if len(embedding) > max_amplitudes:
        # Truncate if too large
        processed_embedding = embedding[:max_amplitudes]
        logger.warning(f"Embedding truncated from {len(embedding)} to {max_amplitudes}")
    else:
        # Pad with zeros if too small
        processed_embedding = np.pad(embedding, (0, max_amplitudes - len(embedding)))
    
    # Normalize to unit vector (required for quantum state)
    norm = np.linalg.norm(processed_embedding)
    if norm > 0:
        processed_embedding = processed_embedding / norm
    
    # Create circuit and initialize with amplitudes
    qc = QuantumCircuit(self.n_qubits, name="amplitude_encoded")
    qc.initialize(processed_embedding, range(self.n_qubits))
    
    # Verify circuit depth doesn't exceed PRD limit
    if qc.depth() > self.max_depth:
        logger.warning(f"Circuit depth {qc.depth()} exceeds limit {self.max_depth}")
    
    return qc

def angle_encode_embedding(self, embedding: np.ndarray) -> QuantumCircuit:
    """
    Encode classical embedding using rotation angles.
    
    Alternative encoding method for comparison.
    """
    qc = QuantumCircuit(self.n_qubits, name="angle_encoded")
    
    # Use first n_qubits values as rotation angles
    angles = embedding[:self.n_qubits]
    
    for i, angle in enumerate(angles):
        qc.ry(angle, i)  # Rotation around Y-axis
    
    return qc
```

### 3. Add Circuit Simulation Capabilities
```python
def simulate_circuit(self, circuit: QuantumCircuit) -> Tuple[Statevector, dict]:
    """
    Simulate quantum circuit and return statevector.
    
    Args:
        circuit: Quantum circuit to simulate
        
    Returns:
        Tuple of (statevector, metadata)
    """
    try:
        # Run simulation
        job = self.simulator.run(circuit)
        result = job.result()
        
        # Get statevector
        statevector = result.get_statevector(circuit)
        
        # Collect metadata
        metadata = {
            'circuit_depth': circuit.depth(),
            'circuit_size': circuit.size(),
            'n_qubits': circuit.num_qubits,
            'success': True
        }
        
        logger.debug(f"Simulated circuit: depth={metadata['circuit_depth']}, size={metadata['circuit_size']}")
        
        return statevector, metadata
    
    except Exception as e:
        logger.error(f"Circuit simulation failed: {e}")
        metadata = {'success': False, 'error': str(e)}
        return None, metadata

def get_circuit_properties(self, circuit: QuantumCircuit) -> dict:
    """Get detailed properties of a quantum circuit."""
    return {
        'name': circuit.name,
        'num_qubits': circuit.num_qubits,
        'depth': circuit.depth(),
        'size': circuit.size(),
        'operations': circuit.count_ops(),
        'parameters': circuit.num_parameters
    }
```

### 4. Create Testing and Validation Functions
```python
def validate_circuit_constraints(self, circuit: QuantumCircuit) -> bool:
    """
    Validate circuit meets PRD constraints.
    
    Returns:
        True if circuit meets all PRD requirements
    """
    constraints_met = True
    issues = []
    
    # Check qubit count (PRD: 2-4 qubits)
    if not 2 <= circuit.num_qubits <= 4:
        constraints_met = False
        issues.append(f"Qubit count {circuit.num_qubits} outside range [2,4]")
    
    # Check circuit depth (PRD: ≤15 gates)
    if circuit.depth() > self.max_depth:
        constraints_met = False
        issues.append(f"Circuit depth {circuit.depth()} exceeds limit {self.max_depth}")
    
    # Log issues if any
    if issues:
        logger.warning(f"Circuit constraint violations: {issues}")
    
    return constraints_met

def benchmark_simulation_performance(self) -> dict:
    """
    Benchmark simulation performance for different circuit types.
    
    Returns performance metrics aligned with PRD targets.
    """
    import time
    
    results = {}
    
    # Test different circuit types
    test_circuits = [
        ("empty", self.create_empty_circuit()),
        ("superposition", self.create_superposition_circuit()),
        ("entanglement", self.create_entanglement_circuit())
    ]
    
    for name, circuit in test_circuits:
        start_time = time.time()
        statevector, metadata = self.simulate_circuit(circuit)
        simulation_time = time.time() - start_time
        
        results[name] = {
            'simulation_time_ms': simulation_time * 1000,
            'circuit_depth': circuit.depth(),
            'circuit_size': circuit.size(),
            'success': metadata['success'] if metadata else False
        }
        
        logger.info(f"Circuit {name}: {simulation_time*1000:.2f}ms")
    
    return results
```

## Success Criteria

### Functional Requirements
- [ ] Can create quantum circuits with 2-4 qubits
- [ ] Circuit depth stays ≤15 gates as per PRD
- [ ] Amplitude encoding works for embedding vectors
- [ ] Circuit simulation returns valid statevectors
- [ ] All circuits validate against PRD constraints

### Performance Requirements
- [ ] Circuit creation time <10ms
- [ ] Simulation time <100ms (supporting PRD similarity target)
- [ ] Memory usage reasonable for target embedding sizes

### Integration Requirements
- [ ] Clean interface for other modules to use
- [ ] Proper error handling and logging
- [ ] Compatible with embedding dimensions from SentenceTransformers

## Files to Create
```
quantum_rerank/core/
├── __init__.py
├── quantum_circuits.py
└── circuit_validators.py

tests/unit/
├── test_quantum_circuits.py
└── test_circuit_validation.py

examples/
└── basic_circuit_demo.py
```

## Testing & Validation

### Unit Tests
```python
# tests/unit/test_quantum_circuits.py
def test_circuit_creation():
    qc_handler = BasicQuantumCircuits(n_qubits=4)
    circuit = qc_handler.create_empty_circuit()
    assert circuit.num_qubits == 4

def test_amplitude_encoding():
    qc_handler = BasicQuantumCircuits(n_qubits=4)
    embedding = np.random.random(16)  # 2^4 = 16 amplitudes
    circuit = qc_handler.amplitude_encode_embedding(embedding)
    assert qc_handler.validate_circuit_constraints(circuit)

def test_simulation_performance():
    qc_handler = BasicQuantumCircuits(n_qubits=4)
    results = qc_handler.benchmark_simulation_performance()
    
    # Check PRD performance targets
    for circuit_type, metrics in results.items():
        assert metrics['simulation_time_ms'] < 100  # Support PRD similarity target
        assert metrics['success']
```

### Integration Test
```python
# examples/basic_circuit_demo.py
def demo_quantum_circuits():
    """Demonstrate basic quantum circuit functionality."""
    qc_handler = BasicQuantumCircuits(n_qubits=4)
    
    # Test different circuit types
    circuits = [
        qc_handler.create_superposition_circuit(),
        qc_handler.create_entanglement_circuit()
    ]
    
    for circuit in circuits:
        # Validate constraints
        is_valid = qc_handler.validate_circuit_constraints(circuit)
        print(f"Circuit {circuit.name}: valid={is_valid}")
        
        # Simulate
        statevector, metadata = qc_handler.simulate_circuit(circuit)
        print(f"Simulation: {metadata}")
        
        # Show properties
        props = qc_handler.get_circuit_properties(circuit)
        print(f"Properties: {props}")
```

## Next Task Dependencies
This task enables:
- Task 03: SentenceTransformer Integration (needs amplitude encoding)
- Task 04: SWAP Test Implementation (needs circuit simulation)
- Task 05: Quantum Parameter Prediction (needs circuit creation)

## References
- PRD Section 1.3: Proven Implementation Patterns
- PRD Section 4.1: System Requirements
- Documentation: Qiskit implementation guide
- Research Papers: Quantum encoding techniques