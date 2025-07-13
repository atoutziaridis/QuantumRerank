"""
Unit tests for SWAP test implementation.

Tests the QuantumSWAPTest class and related functionality for
Task 04: SWAP Test Implementation for Quantum Fidelity.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_rerank.core.swap_test import QuantumSWAPTest, SWAPTestConfig


class TestSWAPTestConfig:
    """Test SWAPTestConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SWAPTestConfig()
        
        assert config.shots == 1024
        assert config.simulator_method == 'statevector'
        assert config.measurement_basis == 'computational'
        assert config.error_mitigation is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SWAPTestConfig(
            shots=2048,
            simulator_method='qasm',
            error_mitigation=True
        )
        
        assert config.shots == 2048
        assert config.simulator_method == 'qasm'
        assert config.error_mitigation is True


class TestQuantumSWAPTest:
    """Test QuantumSWAPTest class."""
    
    def test_init_valid_qubits(self):
        """Test initialization with valid qubit counts."""
        # Test valid qubit counts (2-4)
        for n_qubits in [2, 3, 4]:
            swap_test = QuantumSWAPTest(n_qubits=n_qubits)
            assert swap_test.n_qubits == n_qubits
            assert swap_test.total_qubits == 2 * n_qubits + 1
    
    def test_init_invalid_qubits(self):
        """Test initialization with invalid qubit counts."""
        # Test invalid qubit counts
        with pytest.raises(ValueError, match="n_qubits must be between 2 and 4"):
            QuantumSWAPTest(n_qubits=1)
        
        with pytest.raises(ValueError, match="n_qubits must be between 2 and 4"):
            QuantumSWAPTest(n_qubits=5)
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = SWAPTestConfig(shots=512, simulator_method='qasm')
        swap_test = QuantumSWAPTest(n_qubits=3, config=config)
        
        assert swap_test.config.shots == 512
        assert swap_test.config.simulator_method == 'qasm'
    
    def test_create_swap_test_circuit_valid_inputs(self):
        """Test SWAP test circuit creation with valid inputs."""
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # Create simple test circuits
        circuit1 = QuantumCircuit(2)
        circuit1.h(0)  # |+0⟩
        
        circuit2 = QuantumCircuit(2)
        circuit2.x(1)  # |01⟩
        
        swap_circuit = swap_test.create_swap_test_circuit(circuit1, circuit2)
        
        # Verify circuit structure
        assert swap_circuit.num_qubits == 5  # 2*2 + 1 ancilla
        assert swap_circuit.num_clbits == 1  # One classical bit for measurement
        assert swap_circuit.name == "swap_test"
        
        # Check that circuit has reasonable depth (should be <= 15 per PRD)
        assert swap_circuit.depth() <= 15
    
    def test_create_swap_test_circuit_invalid_inputs(self):
        """Test SWAP test circuit creation with invalid inputs."""
        swap_test = QuantumSWAPTest(n_qubits=3)
        
        # Test with wrong number of qubits
        circuit1 = QuantumCircuit(2)  # Wrong size
        circuit2 = QuantumCircuit(3)  # Correct size
        
        with pytest.raises(ValueError, match="Input circuits must have 3 qubits"):
            swap_test.create_swap_test_circuit(circuit1, circuit2)
    
    def test_compute_fidelity_from_counts_identical_states(self):
        """Test fidelity computation for identical states."""
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # For identical states, P(|0⟩) should be close to 1
        # Therefore fidelity should be close to 1
        counts = {'0': 900, '1': 100}  # Mostly |0⟩ measurements
        
        fidelity = swap_test.compute_fidelity_from_counts(counts)
        
        # F² = 2 * P(0) - 1 = 2 * 0.9 - 1 = 0.8, so F = sqrt(0.8) ≈ 0.894
        expected_fidelity = np.sqrt(2 * 0.9 - 1)
        assert np.isclose(fidelity, expected_fidelity, atol=0.01)
    
    def test_compute_fidelity_from_counts_orthogonal_states(self):
        """Test fidelity computation for orthogonal states."""
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # For orthogonal states, P(|0⟩) should be close to 0.5
        # Therefore fidelity should be close to 0
        counts = {'0': 500, '1': 500}  # Equal measurements
        
        fidelity = swap_test.compute_fidelity_from_counts(counts)
        
        # F² = 2 * 0.5 - 1 = 0, so F = 0
        assert np.isclose(fidelity, 0.0, atol=0.1)
    
    def test_compute_fidelity_from_counts_edge_cases(self):
        """Test fidelity computation edge cases."""
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # Test case where P(0) = 0 (should give fidelity = 0)
        counts_all_one = {'1': 1000}
        fidelity = swap_test.compute_fidelity_from_counts(counts_all_one)
        assert fidelity == 0.0
        
        # Test case where P(0) = 1 (should give fidelity = 1)  
        counts_all_zero = {'0': 1000}
        fidelity = swap_test.compute_fidelity_from_counts(counts_all_zero)
        assert fidelity == 1.0
    
    @patch('quantum_rerank.core.swap_test.AerSimulator')
    def test_compute_fidelity_success(self, mock_simulator_class):
        """Test successful fidelity computation."""
        # Setup mock simulator
        mock_simulator = Mock()
        mock_job = Mock()
        mock_result = Mock()
        
        mock_simulator_class.return_value = mock_simulator
        mock_simulator.run.return_value = mock_job
        mock_job.result.return_value = mock_result
        mock_result.get_counts.return_value = {'0': 800, '1': 200}
        
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # Create test circuits
        circuit1 = QuantumCircuit(2)
        circuit2 = QuantumCircuit(2)
        
        fidelity, metadata = swap_test.compute_fidelity(circuit1, circuit2)
        
        # Verify fidelity calculation
        # P(0) = 0.8, F² = 2*0.8 - 1 = 0.6, F = sqrt(0.6) ≈ 0.775
        expected_fidelity = np.sqrt(2 * 0.8 - 1)
        assert np.isclose(fidelity, expected_fidelity, atol=0.01)
        
        # Verify metadata
        assert metadata['success'] is True
        assert 'execution_time_ms' in metadata
        assert 'shots' in metadata
        assert 'circuit_depth' in metadata
        assert 'measurement_counts' in metadata
    
    @patch('quantum_rerank.core.swap_test.AerSimulator')
    def test_compute_fidelity_failure(self, mock_simulator_class):
        """Test fidelity computation failure handling."""
        # Setup mock simulator to raise exception
        mock_simulator = Mock()
        mock_simulator_class.return_value = mock_simulator
        mock_simulator.run.side_effect = Exception("Simulation failed")
        
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        circuit1 = QuantumCircuit(2)
        circuit2 = QuantumCircuit(2)
        
        fidelity, metadata = swap_test.compute_fidelity(circuit1, circuit2)
        
        assert fidelity == 0.0
        assert metadata['success'] is False
        assert 'error' in metadata
        assert metadata['error'] == "Simulation failed"
    
    @patch('quantum_rerank.core.swap_test.Statevector')
    def test_compute_fidelity_statevector_success(self, mock_statevector_class):
        """Test statevector-based fidelity computation."""
        # Setup mock statevectors
        mock_sv1 = Mock()
        mock_sv2 = Mock()
        mock_statevector_class.from_instruction.side_effect = [mock_sv1, mock_sv2]
        
        # Mock inner product to return known value
        mock_sv1.inner.return_value = 0.8 + 0.6j  # |inner product|² = 1.0
        
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        circuit1 = QuantumCircuit(2)
        circuit2 = QuantumCircuit(2)
        
        fidelity, metadata = swap_test.compute_fidelity_statevector(circuit1, circuit2)
        
        # Verify fidelity calculation: |0.8 + 0.6j|² = 0.64 + 0.36 = 1.0
        assert np.isclose(fidelity, 1.0, atol=0.01)
        assert metadata['success'] is True
        assert metadata['method'] == 'statevector'
    
    @patch('quantum_rerank.core.swap_test.Statevector')
    def test_compute_fidelity_statevector_failure(self, mock_statevector_class):
        """Test statevector fidelity computation failure."""
        mock_statevector_class.from_instruction.side_effect = Exception("Statevector failed")
        
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        circuit1 = QuantumCircuit(2)
        circuit2 = QuantumCircuit(2)
        
        fidelity, metadata = swap_test.compute_fidelity_statevector(circuit1, circuit2)
        
        assert fidelity == 0.0
        assert metadata['success'] is False
        assert metadata['method'] == 'statevector'
        assert 'error' in metadata
    
    def test_batch_compute_fidelity(self):
        """Test batch fidelity computation."""
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # Mock the compute_fidelity method
        with patch.object(swap_test, 'compute_fidelity') as mock_compute:
            mock_compute.side_effect = [
                (0.9, {'success': True, 'time': 50}),
                (0.7, {'success': True, 'time': 45}),
                (0.5, {'success': True, 'time': 55})
            ]
            
            query_circuit = QuantumCircuit(2)
            candidate_circuits = [QuantumCircuit(2) for _ in range(3)]
            
            results = swap_test.batch_compute_fidelity(query_circuit, candidate_circuits)
            
            assert len(results) == 3
            assert mock_compute.call_count == 3
            
            # Check that batch metadata was added
            for i, (fidelity, metadata) in enumerate(results):
                assert metadata['batch_index'] == i
                assert metadata['total_candidates'] == 3
    
    @patch('quantum_rerank.core.swap_test.BasicQuantumCircuits')
    def test_validate_swap_test_implementation(self, mock_qc_class):
        """Test SWAP test validation."""
        # Setup mock quantum circuits
        mock_qc = Mock()
        mock_qc_class.return_value = mock_qc
        
        # Mock different circuits
        identical_circuit = Mock(spec=QuantumCircuit)
        zero_circuit = Mock(spec=QuantumCircuit)
        one_circuit = Mock(spec=QuantumCircuit)
        super_circuit = Mock(spec=QuantumCircuit)
        entangle_circuit = Mock(spec=QuantumCircuit)
        
        mock_qc.create_superposition_circuit.return_value = super_circuit
        mock_qc.create_empty_circuit.return_value = zero_circuit
        mock_qc.create_entanglement_circuit.return_value = entangle_circuit
        
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # Mock fidelity computations for different test cases
        with patch.object(swap_test, 'compute_fidelity') as mock_compute:
            with patch.object(swap_test, 'compute_fidelity_statevector') as mock_compute_sv:
                # Identical states: fidelity ≈ 1
                # Orthogonal states: fidelity ≈ 0  
                # Different states: some intermediate value
                mock_compute.side_effect = [
                    (0.95, {}),  # identical states
                    (0.05, {}),  # orthogonal states  
                    (0.6, {})    # comparison test
                ]
                mock_compute_sv.return_value = (0.58, {})  # statevector comparison
                
                # Mock QuantumCircuit creation for orthogonal test
                with patch('quantum_rerank.core.swap_test.QuantumCircuit') as mock_qc_constructor:
                    mock_qc_constructor.return_value = one_circuit
                    
                    results = swap_test.validate_swap_test_implementation()
                
                # Check validation results
                assert 'identical_states' in results
                assert 'orthogonal_states' in results
                assert 'method_comparison' in results
                assert 'overall_pass' in results
                
                # Check individual test results
                assert results['identical_states']['pass'] is True  # 0.95 close to 1.0
                assert results['orthogonal_states']['pass'] is True  # 0.05 close to 0.0
                assert results['method_comparison']['pass'] is True  # 0.6 vs 0.58 within tolerance
    
    @patch('quantum_rerank.core.swap_test.BasicQuantumCircuits')
    def test_benchmark_swap_test_performance(self, mock_qc_class):
        """Test SWAP test performance benchmarking."""
        # Setup mock quantum circuits
        mock_qc = Mock()
        mock_qc_class.return_value = mock_qc
        
        test_circuits = [Mock(spec=QuantumCircuit) for _ in range(3)]
        mock_qc.create_empty_circuit.return_value = test_circuits[0]
        mock_qc.create_superposition_circuit.return_value = test_circuits[1]  
        mock_qc.create_entanglement_circuit.return_value = test_circuits[2]
        
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # Mock fidelity computations with known timing
        with patch.object(swap_test, 'compute_fidelity') as mock_compute:
            with patch.object(swap_test, 'batch_compute_fidelity') as mock_batch:
                # Mock individual fidelity calls (3x3 = 9 calls)
                mock_compute.side_effect = [
                    (0.5, {}) for _ in range(9)
                ]
                
                # Mock batch computation
                mock_batch.return_value = [(0.5, {}) for _ in range(3)]
                
                results = swap_test.benchmark_swap_test_performance()
                
                # Check result structure
                assert 'single_fidelity_times' in results
                assert 'batch_processing_time' in results
                assert 'performance_summary' in results
                
                # Check that all pairwise comparisons were done
                assert len(results['single_fidelity_times']) == 9  # 3x3 combinations
                
                # Check performance summary
                summary = results['performance_summary']
                assert 'avg_single_fidelity_ms' in summary
                assert 'max_single_fidelity_ms' in summary
                assert 'meets_prd_target' in summary
                assert 'batch_efficiency' in summary


class TestSWAPTestIntegration:
    """Integration tests for SWAP test with real quantum circuits."""
    
    def test_real_circuit_fidelity_computation(self):
        """Test fidelity computation with real quantum circuits."""
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # Create two simple but different circuits
        circuit1 = QuantumCircuit(2)
        circuit1.h(0)  # |+0⟩ state
        
        circuit2 = QuantumCircuit(2)
        circuit2.h(0)
        circuit2.z(0)  # |-0⟩ state
        
        # These states should be orthogonal, so fidelity should be close to 0
        with patch.object(swap_test.simulator, 'run') as mock_run:
            mock_job = Mock()
            mock_result = Mock()
            mock_run.return_value = mock_job
            mock_job.result.return_value = mock_result
            # Mock equal probability for orthogonal states
            mock_result.get_counts.return_value = {'0': 500, '1': 500}
            
            fidelity, metadata = swap_test.compute_fidelity(circuit1, circuit2)
            
            assert np.isclose(fidelity, 0.0, atol=0.1)
            assert metadata['success'] is True
    
    def test_identical_circuit_fidelity(self):
        """Test fidelity computation with identical circuits."""
        swap_test = QuantumSWAPTest(n_qubits=2)
        
        # Create identical circuits
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)  # Bell state
        
        # Identical circuits should have fidelity close to 1
        with patch.object(swap_test.simulator, 'run') as mock_run:
            mock_job = Mock()
            mock_result = Mock()
            mock_run.return_value = mock_job
            mock_job.result.return_value = mock_result
            # Mock high probability of |0⟩ for identical states
            mock_result.get_counts.return_value = {'0': 950, '1': 50}
            
            fidelity, metadata = swap_test.compute_fidelity(circuit, circuit)
            
            # Should be close to 1.0
            assert fidelity > 0.8
            assert metadata['success'] is True