"""
Unit tests for quantum circuit creation and simulation.

Tests basic quantum circuit functionality against PRD requirements:
- 2-4 qubits constraint validation
- ≤15 gate depth constraint validation
- Amplitude and angle encoding functionality
- Performance targets (<100ms simulation)
"""

import pytest
import numpy as np
import time
from unittest.mock import patch

from quantum_rerank.core.quantum_circuits import (
    BasicQuantumCircuits,
    CircuitResult,
    CircuitProperties
)
from quantum_rerank.config.settings import QuantumConfig


class TestBasicQuantumCircuits:
    """Test basic quantum circuit creation and simulation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = QuantumConfig(n_qubits=4, max_circuit_depth=15)
        self.circuit_handler = BasicQuantumCircuits(self.config)
    
    def test_initialization_valid_config(self):
        """Test circuit handler initialization with valid configuration."""
        handler = BasicQuantumCircuits(self.config)
        
        assert handler.n_qubits == 4
        assert handler.max_depth == 15
        assert handler.shots == 1024
        assert handler.simulator is not None
    
    def test_initialization_invalid_qubits(self):
        """Test circuit handler initialization with invalid qubit count."""
        # Test too few qubits
        with pytest.raises(ValueError, match="n_qubits must be between 2 and 4"):
            config = QuantumConfig(n_qubits=1)
            BasicQuantumCircuits(config)
        
        # Test too many qubits
        with pytest.raises(ValueError, match="n_qubits must be between 2 and 4"):
            config = QuantumConfig(n_qubits=5)
            BasicQuantumCircuits(config)
    
    def test_initialization_invalid_depth(self):
        """Test circuit handler initialization with invalid depth."""
        with pytest.raises(ValueError, match="max_circuit_depth must be ≤15"):
            config = QuantumConfig(n_qubits=4, max_circuit_depth=20)
            BasicQuantumCircuits(config)
    
    def test_create_empty_circuit(self):
        """Test creation of empty quantum circuit."""
        circuit = self.circuit_handler.create_empty_circuit("test_empty")
        
        assert circuit.name == "test_empty"
        assert circuit.num_qubits == 4
        assert circuit.depth() == 0
        assert circuit.size() == 0
    
    def test_create_superposition_circuit(self):
        """Test creation of superposition circuit."""
        circuit = self.circuit_handler.create_superposition_circuit()
        
        assert circuit.name == "superposition"
        assert circuit.num_qubits == 4
        assert circuit.depth() == 1  # All H gates in parallel
        assert self.circuit_handler.validate_circuit_constraints(circuit)
    
    def test_create_entanglement_circuit(self):
        """Test creation of entanglement circuit."""
        circuit = self.circuit_handler.create_entanglement_circuit()
        
        assert circuit.name == "entanglement"
        assert circuit.num_qubits == 4
        assert circuit.depth() >= 1
        assert self.circuit_handler.validate_circuit_constraints(circuit)
    
    def test_amplitude_encode_embedding_valid(self):
        """Test amplitude encoding with valid embedding."""
        # Test with embedding that fits exactly
        embedding = np.random.rand(16)  # 2^4 = 16 amplitudes
        circuit = self.circuit_handler.amplitude_encode_embedding(embedding)
        
        assert circuit.name == "amplitude_encoded"
        assert circuit.num_qubits == 4
        assert self.circuit_handler.validate_circuit_constraints(circuit)
    
    def test_amplitude_encode_embedding_truncation(self):
        """Test amplitude encoding with oversized embedding."""
        # Test with embedding larger than circuit capacity
        embedding = np.random.rand(32)  # Larger than 2^4 = 16
        
        with pytest.warns(None):  # Should log warning about truncation
            circuit = self.circuit_handler.amplitude_encode_embedding(embedding)
        
        assert circuit.num_qubits == 4
        assert self.circuit_handler.validate_circuit_constraints(circuit)
    
    def test_amplitude_encode_embedding_padding(self):
        """Test amplitude encoding with small embedding."""
        # Test with embedding smaller than circuit capacity
        embedding = np.random.rand(8)  # Smaller than 2^4 = 16
        circuit = self.circuit_handler.amplitude_encode_embedding(embedding)
        
        assert circuit.num_qubits == 4
        assert self.circuit_handler.validate_circuit_constraints(circuit)
    
    def test_amplitude_encode_zero_embedding(self):
        """Test amplitude encoding with zero embedding."""
        embedding = np.zeros(16)
        
        with pytest.raises(ValueError, match="Cannot encode zero embedding vector"):
            self.circuit_handler.amplitude_encode_embedding(embedding)
    
    def test_angle_encode_embedding_valid(self):
        """Test angle encoding with valid parameters."""
        embedding = np.random.rand(10)
        
        # Test different rotation types
        for encoding_type in ['rx', 'ry', 'rz']:
            circuit = self.circuit_handler.angle_encode_embedding(
                embedding, encoding_type=encoding_type
            )
            
            assert circuit.name == "angle_encoded"
            assert circuit.num_qubits == 4
            assert self.circuit_handler.validate_circuit_constraints(circuit)
    
    def test_angle_encode_invalid_type(self):
        """Test angle encoding with invalid rotation type."""
        embedding = np.random.rand(4)
        
        with pytest.raises(ValueError, match="Invalid encoding_type"):
            self.circuit_handler.angle_encode_embedding(
                embedding, encoding_type='invalid'
            )
    
    def test_dense_angle_encoding(self):
        """Test dense angle encoding."""
        embedding = np.random.rand(10)
        circuit = self.circuit_handler.dense_angle_encoding(embedding)
        
        assert circuit.name == "dense_angle_encoded"
        assert circuit.num_qubits == 4
        # Dense encoding may have higher depth due to entangling gates
        assert circuit.depth() >= 1
    
    def test_simulate_circuit_success(self):
        """Test successful circuit simulation."""
        circuit = self.circuit_handler.create_superposition_circuit()
        result = self.circuit_handler.simulate_circuit(circuit)
        
        assert isinstance(result, CircuitResult)
        assert result.success
        assert result.statevector is not None
        assert result.metadata is not None
        assert result.metadata['circuit_name'] == 'superposition'
        assert 'simulation_time_ms' in result.metadata
        assert result.metadata['prd_compliant'] is True
    
    def test_simulate_circuit_performance(self):
        """Test circuit simulation meets performance targets."""
        circuit = self.circuit_handler.create_superposition_circuit()
        
        start_time = time.time()
        result = self.circuit_handler.simulate_circuit(circuit)
        simulation_time = (time.time() - start_time) * 1000
        
        # PRD target: <100ms simulation supporting similarity computation
        assert simulation_time < 100
        assert result.success
        assert result.metadata['simulation_time_ms'] < 100
    
    def test_get_circuit_properties(self):
        """Test circuit property analysis."""
        circuit = self.circuit_handler.create_entanglement_circuit()
        properties = self.circuit_handler.get_circuit_properties(circuit)
        
        assert isinstance(properties, CircuitProperties)
        assert properties.name == "entanglement"
        assert properties.num_qubits == 4
        assert properties.depth >= 1
        assert properties.size >= 4  # At least H + CNOTs
        assert isinstance(properties.operations, dict)
        assert properties.prd_compliant is True
    
    def test_validate_circuit_constraints_valid(self):
        """Test validation of valid circuits."""
        circuit = self.circuit_handler.create_superposition_circuit()
        is_valid = self.circuit_handler.validate_circuit_constraints(circuit)
        
        assert is_valid is True
    
    def test_validate_circuit_constraints_invalid_qubits(self):
        """Test validation with invalid qubit count."""
        # Create circuit with wrong number of qubits
        from qiskit import QuantumCircuit
        invalid_circuit = QuantumCircuit(1)  # Too few qubits
        
        is_valid = self.circuit_handler.validate_circuit_constraints(invalid_circuit)
        assert is_valid is False
    
    def test_benchmark_simulation_performance(self):
        """Test performance benchmarking functionality."""
        results = self.circuit_handler.benchmark_simulation_performance(num_trials=3)
        
        assert isinstance(results, dict)
        assert 'summary' in results
        assert 'overall_avg_time_ms' in results['summary']
        assert 'overall_prd_compliant' in results['summary']
        
        # Check individual circuit results
        expected_circuits = ['empty', 'superposition', 'entanglement', 
                           'amplitude_encoding', 'angle_encoding', 'dense_angle_encoding']
        
        for circuit_name in expected_circuits:
            assert circuit_name in results
            assert 'avg_simulation_time_ms' in results[circuit_name]
            assert 'prd_compliant' in results[circuit_name]
            
            # All should meet PRD target
            assert results[circuit_name]['avg_simulation_time_ms'] < 100
    
    def test_compare_encoding_methods(self):
        """Test comparison of different encoding methods."""
        # Create test embeddings
        embeddings = [np.random.rand(16) for _ in range(3)]
        
        results = self.circuit_handler.compare_encoding_methods(embeddings)
        
        assert isinstance(results, dict)
        expected_methods = ['amplitude', 'angle', 'dense_angle']
        
        for method in expected_methods:
            assert method in results
            assert 'avg_encoding_time_ms' in results[method]
            assert 'avg_simulation_time_ms' in results[method]
            assert 'total_time_ms' in results[method]
            assert 'prd_compliant' in results[method]
            
            # Should meet PRD targets
            assert results[method]['total_time_ms'] < 100


class TestCircuitResultDataclass:
    """Test CircuitResult dataclass functionality."""
    
    def test_circuit_result_success(self):
        """Test successful circuit result creation."""
        from qiskit.quantum_info import Statevector
        
        statevector = Statevector.from_label('00')
        metadata = {'test': 'data'}
        
        result = CircuitResult(
            success=True,
            statevector=statevector,
            metadata=metadata
        )
        
        assert result.success is True
        assert result.statevector == statevector
        assert result.metadata == metadata
        assert result.error is None
    
    def test_circuit_result_failure(self):
        """Test failed circuit result creation."""
        error_msg = "Test error"
        
        result = CircuitResult(
            success=False,
            error=error_msg
        )
        
        assert result.success is False
        assert result.statevector is None
        assert result.error == error_msg


class TestCircuitPropertiesDataclass:
    """Test CircuitProperties dataclass functionality."""
    
    def test_circuit_properties_creation(self):
        """Test circuit properties dataclass creation."""
        properties = CircuitProperties(
            name="test_circuit",
            num_qubits=4,
            depth=5,
            size=10,
            operations={'h': 4, 'cnot': 3},
            parameters=0,
            prd_compliant=True
        )
        
        assert properties.name == "test_circuit"
        assert properties.num_qubits == 4
        assert properties.depth == 5
        assert properties.size == 10
        assert properties.operations == {'h': 4, 'cnot': 3}
        assert properties.parameters == 0
        assert properties.prd_compliant is True


class TestQuantumConfigIntegration:
    """Test integration with QuantumConfig."""
    
    def test_custom_config_integration(self):
        """Test circuit handler with custom configuration."""
        custom_config = QuantumConfig(
            n_qubits=3,
            max_circuit_depth=10,
            shots=2048,
            simulator_method='statevector'
        )
        
        handler = BasicQuantumCircuits(custom_config)
        
        assert handler.n_qubits == 3
        assert handler.max_depth == 10
        assert handler.shots == 2048
    
    def test_default_config_integration(self):
        """Test circuit handler with default configuration."""
        handler = BasicQuantumCircuits()
        
        # Should use default QuantumConfig values
        default_config = QuantumConfig()
        assert handler.n_qubits == default_config.n_qubits
        assert handler.max_depth == default_config.max_circuit_depth
        assert handler.shots == default_config.shots


class TestPerformanceRequirements:
    """Test performance requirements against PRD targets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.circuit_handler = BasicQuantumCircuits()
    
    def test_simulation_time_target(self):
        """Test that simulation meets <100ms target."""
        circuits = [
            self.circuit_handler.create_empty_circuit(),
            self.circuit_handler.create_superposition_circuit(),
            self.circuit_handler.create_entanglement_circuit()
        ]
        
        for circuit in circuits:
            start_time = time.time()
            result = self.circuit_handler.simulate_circuit(circuit)
            simulation_time = (time.time() - start_time) * 1000
            
            # PRD requirement: <100ms simulation
            assert simulation_time < 100, f"Circuit '{circuit.name}' took {simulation_time:.2f}ms"
            assert result.success, f"Circuit '{circuit.name}' simulation failed"
    
    def test_encoding_performance_target(self):
        """Test that encoding meets performance targets."""
        embedding = np.random.rand(16)
        
        # Test amplitude encoding
        start_time = time.time()
        amplitude_circuit = self.circuit_handler.amplitude_encode_embedding(embedding)
        amplitude_time = (time.time() - start_time) * 1000
        
        # Test angle encoding
        start_time = time.time()
        angle_circuit = self.circuit_handler.angle_encode_embedding(embedding)
        angle_time = (time.time() - start_time) * 1000
        
        # Encoding should be very fast (<10ms)
        assert amplitude_time < 10, f"Amplitude encoding took {amplitude_time:.2f}ms"
        assert angle_time < 10, f"Angle encoding took {angle_time:.2f}ms"
    
    def test_prd_constraint_compliance(self):
        """Test that all circuits comply with PRD constraints."""
        # Test various circuit types
        circuits = [
            self.circuit_handler.create_empty_circuit(),
            self.circuit_handler.create_superposition_circuit(),
            self.circuit_handler.create_entanglement_circuit(),
            self.circuit_handler.amplitude_encode_embedding(np.random.rand(16)),
            self.circuit_handler.angle_encode_embedding(np.random.rand(8))
        ]
        
        for circuit in circuits:
            # PRD constraints
            assert 2 <= circuit.num_qubits <= 4, f"Circuit '{circuit.name}' has {circuit.num_qubits} qubits"
            assert circuit.depth() <= 15, f"Circuit '{circuit.name}' has depth {circuit.depth()}"
            assert self.circuit_handler.validate_circuit_constraints(circuit), f"Circuit '{circuit.name}' violates constraints"


# Integration test markers
@pytest.mark.integration
class TestQuantumCircuitIntegration:
    """Integration tests for quantum circuit functionality."""
    
    def test_end_to_end_embedding_workflow(self):
        """Test complete workflow from embedding to simulation."""
        handler = BasicQuantumCircuits()
        
        # Simulate realistic embedding from sentence transformer
        embedding = np.random.rand(384)  # Typical sentence transformer dimension
        
        # Amplitude encoding workflow
        circuit = handler.amplitude_encode_embedding(embedding)
        result = handler.simulate_circuit(circuit)
        properties = handler.get_circuit_properties(circuit)
        
        assert result.success
        assert properties.prd_compliant
        assert result.metadata['simulation_time_ms'] < 100
    
    def test_performance_benchmark_integration(self):
        """Test performance benchmarking integration."""
        handler = BasicQuantumCircuits()
        
        # Run comprehensive performance benchmark
        results = handler.benchmark_simulation_performance(num_trials=5)
        
        # Should meet PRD requirements
        assert results['summary']['overall_prd_compliant']
        assert results['summary']['overall_avg_time_ms'] < 100
        assert results['summary']['overall_success_rate'] >= 0.95