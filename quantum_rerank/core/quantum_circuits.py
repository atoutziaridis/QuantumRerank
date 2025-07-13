"""
Basic quantum circuit creation and simulation for QuantumRerank.

This module implements small-scale quantum circuits (2-4 qubits, ≤15 gates) 
as specified in the PRD, with amplitude and angle encoding for classical embeddings.

Based on:
- PRD Section 4.1: System Requirements (2-4 qubits, ≤15 gates)
- PRD Section 3.1: Core Algorithms - Amplitude Encoding
- Research: Quantum-inspired embedding techniques
"""

import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RealAmplitudes

from ..config.settings import QuantumConfig

logger = logging.getLogger(__name__)


@dataclass
class CircuitResult:
    """Result container for quantum circuit operations."""
    success: bool
    statevector: Optional[Statevector] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class CircuitProperties:
    """Properties of a quantum circuit."""
    name: str
    num_qubits: int
    depth: int
    size: int
    operations: Dict[str, int]
    parameters: int
    prd_compliant: bool


class BasicQuantumCircuits:
    """
    Basic quantum circuit operations for QuantumRerank.
    
    Implements small-scale circuits as specified in PRD Section 4.1:
    - 2-4 qubits maximum
    - ≤15 gate depth
    - Classical simulation only
    """
    
    def __init__(self, config: Optional[QuantumConfig] = None):
        """
        Initialize quantum circuit handler.
        
        Args:
            config: Quantum configuration (uses default if None)
        """
        self.config = config or QuantumConfig()
        
        # Validate PRD constraints
        if not 2 <= self.config.n_qubits <= 4:
            raise ValueError(f"n_qubits must be between 2 and 4 (PRD requirement), got {self.config.n_qubits}")
        
        if self.config.max_circuit_depth > 15:
            raise ValueError(f"max_circuit_depth must be ≤15 (PRD requirement), got {self.config.max_circuit_depth}")
        
        self.n_qubits = self.config.n_qubits
        self.max_depth = self.config.max_circuit_depth
        self.shots = self.config.shots
        
        # Initialize simulator for classical simulation
        self.simulator = AerSimulator(method=self.config.simulator_method)
        
        logger.info(f"Initialized BasicQuantumCircuits: {self.n_qubits} qubits, depth ≤{self.max_depth}")
    
    def create_empty_circuit(self, name: str = "empty_circuit") -> QuantumCircuit:
        """
        Create an empty quantum circuit with specified qubits.
        
        Args:
            name: Circuit name
            
        Returns:
            Empty quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits, name=name)
        logger.debug(f"Created empty circuit '{name}' with {self.n_qubits} qubits")
        return qc
    
    def create_superposition_circuit(self) -> QuantumCircuit:
        """
        Create a simple superposition circuit for testing.
        
        Returns:
            Quantum circuit with all qubits in superposition
        """
        qc = QuantumCircuit(self.n_qubits, name="superposition")
        
        # Apply Hadamard to all qubits
        for qubit in range(self.n_qubits):
            qc.h(qubit)
        
        logger.debug(f"Created superposition circuit with depth {qc.depth()}")
        return qc
    
    def create_entanglement_circuit(self) -> QuantumCircuit:
        """
        Create a basic entanglement circuit.
        
        Returns:
            Quantum circuit with entangled qubits
        """
        qc = QuantumCircuit(self.n_qubits, name="entanglement")
        
        # Create entanglement between adjacent qubits
        qc.h(0)  # Superposition on first qubit
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)  # Entangle adjacent qubits
        
        logger.debug(f"Created entanglement circuit with depth {qc.depth()}")
        return qc
    
    def amplitude_encode_embedding(self, embedding: np.ndarray, name: str = "amplitude_encoded") -> QuantumCircuit:
        """
        Encode classical embedding into quantum state via amplitude encoding.
        
        Implementation based on PRD Section 3.1 and quantum-inspired research.
        Embeddings are normalized and mapped to quantum state amplitudes.
        
        Args:
            embedding: Classical embedding vector to encode
            name: Circuit name
            
        Returns:
            Quantum circuit with amplitude-encoded state
            
        Raises:
            ValueError: If embedding cannot be properly encoded
        """
        # Calculate maximum amplitudes for current qubit configuration
        max_amplitudes = 2 ** self.n_qubits
        
        # Preprocess embedding to fit quantum circuit
        if len(embedding) > max_amplitudes:
            # Truncate if too large
            processed_embedding = embedding[:max_amplitudes]
            logger.warning(f"Embedding truncated from {len(embedding)} to {max_amplitudes} dimensions")
        else:
            # Pad with zeros if too small
            processed_embedding = np.pad(embedding, (0, max_amplitudes - len(embedding)))
        
        # Normalize to unit vector (required for quantum state)
        norm = np.linalg.norm(processed_embedding)
        if norm == 0:
            raise ValueError("Cannot encode zero embedding vector")
        
        processed_embedding = processed_embedding / norm
        
        # Create circuit and initialize with amplitudes
        qc = QuantumCircuit(self.n_qubits, name=name)
        qc.initialize(processed_embedding, range(self.n_qubits))
        
        # Verify circuit meets PRD constraints
        if qc.depth() > self.max_depth:
            logger.warning(f"Amplitude encoding circuit depth {qc.depth()} exceeds PRD limit {self.max_depth}")
        
        logger.debug(f"Amplitude encoded embedding: {len(embedding)} -> {len(processed_embedding)} amplitudes")
        return qc
    
    def angle_encode_embedding(self, embedding: np.ndarray, 
                             encoding_type: str = 'ry', 
                             name: str = "angle_encoded") -> QuantumCircuit:
        """
        Encode classical embedding using rotation angles.
        
        Alternative encoding method for comparison with amplitude encoding.
        Uses embedding values as rotation angles for quantum gates.
        
        Args:
            embedding: Classical embedding vector to encode
            encoding_type: Type of rotation ('rx', 'ry', 'rz')
            name: Circuit name
            
        Returns:
            Quantum circuit with angle-encoded state
            
        Raises:
            ValueError: If encoding_type is invalid
        """
        if encoding_type not in ['rx', 'ry', 'rz']:
            raise ValueError(f"Invalid encoding_type '{encoding_type}'. Must be 'rx', 'ry', or 'rz'")
        
        qc = QuantumCircuit(self.n_qubits, name=name)
        
        # Use first n_qubits values as rotation angles
        angles = embedding[:self.n_qubits]
        
        for i, angle in enumerate(angles):
            if encoding_type == 'rx':
                qc.rx(angle, i)
            elif encoding_type == 'ry':
                qc.ry(angle, i)
            elif encoding_type == 'rz':
                qc.rz(angle, i)
        
        logger.debug(f"Angle encoded embedding using {encoding_type} rotations")
        return qc
    
    def dense_angle_encoding(self, embedding: np.ndarray, name: str = "dense_angle_encoded") -> QuantumCircuit:
        """
        Create dense angle encoding circuit with multiple rotation layers.
        
        Uses more embedding dimensions by applying multiple rotation layers.
        Based on quantum-inspired compression techniques from research.
        
        Args:
            embedding: Classical embedding vector to encode
            name: Circuit name
            
        Returns:
            Quantum circuit with dense angle encoding
        """
        # Use up to 2 * n_qubits dimensions for dense encoding
        max_dims = 2 * self.n_qubits
        embedding_slice = embedding[:max_dims]
        
        qc = QuantumCircuit(self.n_qubits, name=name)
        
        # First layer: RY rotations
        for i in range(self.n_qubits):
            if i < len(embedding_slice):
                qc.ry(embedding_slice[i], i)
        
        # Second layer: RZ rotations (if we have enough dimensions)
        for i in range(self.n_qubits):
            if i + self.n_qubits < len(embedding_slice):
                qc.rz(embedding_slice[i + self.n_qubits], i)
        
        # Add entangling gates to mix information
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        logger.debug(f"Dense angle encoded {min(len(embedding), max_dims)} dimensions")
        return qc
    
    def simulate_circuit(self, circuit: QuantumCircuit) -> CircuitResult:
        """
        Simulate quantum circuit and return statevector.
        
        Args:
            circuit: Quantum circuit to simulate
            
        Returns:
            CircuitResult with statevector and metadata
        """
        start_time = time.time()
        
        try:
            # Run simulation
            job = self.simulator.run(circuit, shots=1)  # Statevector only needs 1 shot
            result = job.result()
            
            # Get statevector
            statevector = result.get_statevector(circuit)
            
            simulation_time = time.time() - start_time
            
            # Collect metadata
            metadata = {
                'circuit_name': circuit.name,
                'circuit_depth': circuit.depth(),
                'circuit_size': circuit.size(),
                'n_qubits': circuit.num_qubits,
                'simulation_time_ms': simulation_time * 1000,
                'prd_compliant': self.validate_circuit_constraints(circuit),
                'statevector_norm': float(np.abs(statevector.data).sum()),
                'timestamp': time.time()
            }
            
            logger.debug(f"Simulated circuit '{circuit.name}': {simulation_time*1000:.2f}ms")
            
            return CircuitResult(
                success=True,
                statevector=statevector,
                metadata=metadata
            )
        
        except Exception as e:
            simulation_time = time.time() - start_time
            error_msg = f"Circuit simulation failed: {str(e)}"
            logger.error(error_msg)
            
            return CircuitResult(
                success=False,
                metadata={
                    'circuit_name': circuit.name,
                    'simulation_time_ms': simulation_time * 1000,
                    'error': error_msg
                },
                error=error_msg
            )
    
    def get_circuit_properties(self, circuit: QuantumCircuit) -> CircuitProperties:
        """
        Get detailed properties of a quantum circuit.
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            CircuitProperties with detailed analysis
        """
        prd_compliant = self.validate_circuit_constraints(circuit)
        
        return CircuitProperties(
            name=circuit.name,
            num_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            size=circuit.size(),
            operations=circuit.count_ops(),
            parameters=circuit.num_parameters,
            prd_compliant=prd_compliant
        )
    
    def validate_circuit_constraints(self, circuit: QuantumCircuit) -> bool:
        """
        Validate circuit meets PRD constraints.
        
        Args:
            circuit: Quantum circuit to validate
            
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
            logger.warning(f"Circuit '{circuit.name}' constraint violations: {issues}")
        
        return constraints_met
    
    def benchmark_simulation_performance(self, num_trials: int = 10) -> Dict[str, Any]:
        """
        Benchmark simulation performance for different circuit types.
        
        Tests performance against PRD targets:
        - <100ms simulation time (supporting similarity computation target)
        
        Args:
            num_trials: Number of benchmark trials per circuit type
            
        Returns:
            Performance metrics aligned with PRD targets
        """
        logger.info(f"Running simulation performance benchmark with {num_trials} trials")
        
        results = {}
        
        # Test different circuit types
        test_circuits = [
            ("empty", self.create_empty_circuit()),
            ("superposition", self.create_superposition_circuit()),
            ("entanglement", self.create_entanglement_circuit())
        ]
        
        # Add encoding circuits with sample data
        sample_embedding = np.random.rand(16)  # Sample 16-dimensional embedding
        test_circuits.extend([
            ("amplitude_encoding", self.amplitude_encode_embedding(sample_embedding)),
            ("angle_encoding", self.angle_encode_embedding(sample_embedding)),
            ("dense_angle_encoding", self.dense_angle_encoding(sample_embedding))
        ])
        
        for name, circuit in test_circuits:
            trial_times = []
            trial_success = []
            
            for trial in range(num_trials):
                result = self.simulate_circuit(circuit)
                trial_times.append(result.metadata['simulation_time_ms'])
                trial_success.append(result.success)
            
            # Calculate statistics
            avg_time = np.mean(trial_times)
            min_time = np.min(trial_times)
            max_time = np.max(trial_times)
            std_time = np.std(trial_times)
            success_rate = np.mean(trial_success)
            
            # Check PRD compliance
            prd_compliant = avg_time < 100  # PRD target: <100ms for similarity computation
            
            results[name] = {
                'avg_simulation_time_ms': avg_time,
                'min_simulation_time_ms': min_time,
                'max_simulation_time_ms': max_time,
                'std_simulation_time_ms': std_time,
                'success_rate': success_rate,
                'circuit_depth': circuit.depth(),
                'circuit_size': circuit.size(),
                'prd_compliant': prd_compliant,
                'num_trials': num_trials
            }
            
            logger.info(f"Circuit '{name}': {avg_time:.2f}ms avg, PRD compliant: {prd_compliant}")
        
        # Overall summary
        overall_avg = np.mean([r['avg_simulation_time_ms'] for r in results.values()])
        overall_success = np.mean([r['success_rate'] for r in results.values()])
        
        results['summary'] = {
            'overall_avg_time_ms': overall_avg,
            'overall_success_rate': overall_success,
            'prd_target_ms': 100,
            'overall_prd_compliant': overall_avg < 100,
            'benchmark_timestamp': time.time()
        }
        
        logger.info(f"Benchmark complete: {overall_avg:.2f}ms avg, {overall_success*100:.1f}% success")
        
        return results
    
    def compare_encoding_methods(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compare different encoding methods for performance and characteristics.
        
        Args:
            embeddings: List of embedding vectors to test
            
        Returns:
            Comparison results for different encoding methods
        """
        methods = ['amplitude', 'angle', 'dense_angle']
        results = {}
        
        logger.info(f"Comparing encoding methods on {len(embeddings)} embeddings")
        
        for method in methods:
            method_results = {
                'encoding_times': [],
                'simulation_times': [],
                'circuit_depths': [],
                'circuit_sizes': [],
                'success_rates': []
            }
            
            for embedding in embeddings:
                start_time = time.time()
                
                # Create circuit based on method
                if method == 'amplitude':
                    circuit = self.amplitude_encode_embedding(embedding)
                elif method == 'angle':
                    circuit = self.angle_encode_embedding(embedding)
                else:  # dense_angle
                    circuit = self.dense_angle_encoding(embedding)
                
                encoding_time = time.time() - start_time
                
                # Simulate circuit
                sim_result = self.simulate_circuit(circuit)
                
                # Collect metrics
                method_results['encoding_times'].append(encoding_time * 1000)
                method_results['simulation_times'].append(sim_result.metadata['simulation_time_ms'])
                method_results['circuit_depths'].append(circuit.depth())
                method_results['circuit_sizes'].append(circuit.size())
                method_results['success_rates'].append(sim_result.success)
            
            # Calculate summary statistics
            results[method] = {
                'avg_encoding_time_ms': np.mean(method_results['encoding_times']),
                'avg_simulation_time_ms': np.mean(method_results['simulation_times']),
                'avg_circuit_depth': np.mean(method_results['circuit_depths']),
                'avg_circuit_size': np.mean(method_results['circuit_sizes']),
                'success_rate': np.mean(method_results['success_rates']),
                'total_time_ms': np.mean(method_results['encoding_times']) + np.mean(method_results['simulation_times']),
                'prd_compliant': (np.mean(method_results['encoding_times']) + np.mean(method_results['simulation_times'])) < 100
            }
            
            logger.info(f"Method '{method}': {results[method]['total_time_ms']:.2f}ms total")
        
        return results