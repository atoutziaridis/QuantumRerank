"""
Parameterized Quantum Circuits Module.

This module creates parameterized quantum circuits using predicted parameters,
bridging classical parameter prediction with quantum circuit construction.

Based on:
- PRD Section 3.1: Core Algorithms - Parameterized Quantum Circuits (PQC)
- Task 05 specifications for hybrid quantum-classical approach
"""

import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from typing import Dict, Tuple, List, Optional
import logging
import time

from ..core.quantum_circuits import BasicQuantumCircuits
from .parameter_predictor import QuantumParameterPredictor

logger = logging.getLogger(__name__)


class ParameterizedQuantumCircuits:
    """
    Creates parameterized quantum circuits using predicted parameters.
    
    Bridges classical parameter prediction with quantum circuit construction,
    enabling the creation of quantum states that can be used for similarity
    computation via fidelity measurements.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        """
        Initialize parameterized quantum circuit builder.
        
        Args:
            n_qubits: Number of qubits (PRD constraint: 2-4)
            n_layers: Number of circuit layers
        """
        if not 2 <= n_qubits <= 4:
            raise ValueError("n_qubits must be between 2 and 4 (PRD requirement)")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create QuantumConfig for BasicQuantumCircuits
        from ..config.settings import QuantumConfig
        quantum_config = QuantumConfig(n_qubits=n_qubits)
        self.quantum_circuits = BasicQuantumCircuits(quantum_config)
        
        logger.info(f"ParameterizedQuantumCircuits initialized: {n_qubits} qubits, {n_layers} layers")
    
    def create_parameterized_circuit(self, 
                                   parameters: Dict[str, torch.Tensor],
                                   batch_index: int = 0,
                                   circuit_name: str = "parameterized_circuit") -> QuantumCircuit:
        """
        Create a parameterized quantum circuit from predicted parameters.
        
        Creates a PQC with the structure:
        Layer 1: RY-RZ-RY rotations + RZZ entangling gates
        Layer 2: RY-RZ-RY rotations + RZZ entangling gates
        ...
        
        Args:
            parameters: Dictionary of parameter tensors from predictor
            batch_index: Which sample in the batch to use
            circuit_name: Name for the created circuit
            
        Returns:
            Parameterized quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits, name=circuit_name)
        
        # Extract parameters for this sample
        ry_params = parameters['ry_params'][batch_index].detach().cpu().numpy()
        rz_params = parameters['rz_params'][batch_index].detach().cpu().numpy()
        ry2_params = parameters['ry2_params'][batch_index].detach().cpu().numpy()
        entangling_params = parameters['entangling_params'][batch_index].detach().cpu().numpy()
        
        # Build circuit layer by layer
        for layer in range(self.n_layers):
            # Single-qubit rotation gates for each qubit
            for qubit in range(self.n_qubits):
                param_idx = layer * self.n_qubits + qubit
                
                # RY-RZ-RY rotation sequence
                qc.ry(ry_params[param_idx], qubit)
                qc.rz(rz_params[param_idx], qubit)
                qc.ry(ry2_params[param_idx], qubit)
            
            # Two-qubit entangling gates between adjacent qubits
            for qubit in range(self.n_qubits - 1):
                param_idx = layer * (self.n_qubits - 1) + qubit
                qc.rzz(entangling_params[param_idx], qubit, qubit + 1)
        
        # Validate circuit against PRD constraints
        if qc.depth() > 15:  # PRD constraint
            logger.warning(f"Circuit depth {qc.depth()} exceeds PRD limit of 15")
        
        return qc
    
    def create_batch_circuits(self, 
                            parameters: Dict[str, torch.Tensor],
                            circuit_names: Optional[List[str]] = None) -> List[QuantumCircuit]:
        """
        Create multiple parameterized circuits from a batch of parameters.
        
        Args:
            parameters: Batch of parameter tensors
            circuit_names: Optional names for circuits
            
        Returns:
            List of parameterized quantum circuits
        """
        batch_size = parameters['ry_params'].shape[0]
        circuits = []
        
        if circuit_names is None:
            circuit_names = [f"parameterized_circuit_{i}" for i in range(batch_size)]
        
        for i in range(batch_size):
            circuit = self.create_parameterized_circuit(
                parameters, i, circuit_names[i]
            )
            circuits.append(circuit)
        
        logger.debug(f"Created {len(circuits)} parameterized circuits")
        return circuits
    
    def create_embedding_parameterized_circuit(self,
                                             embedding: np.ndarray,
                                             parameters: Dict[str, torch.Tensor],
                                             batch_index: int = 0) -> QuantumCircuit:
        """
        Create a parameterized circuit that includes embedding encoding.
        
        Combines amplitude encoding of the embedding with parameterized gates,
        creating a hybrid classical-quantum state preparation.
        
        Args:
            embedding: Classical embedding to encode
            parameters: Predicted quantum parameters
            batch_index: Batch index for parameters
            
        Returns:
            Quantum circuit with embedding + parameterization
        """
        # First encode the embedding using amplitude encoding
        base_circuit = self.quantum_circuits.amplitude_encode_embedding(
            embedding, name="embedding_parameterized"
        )
        
        # Then apply parameterized gates
        param_circuit = self.create_parameterized_circuit(
            parameters, batch_index, "param_layer"
        )
        
        # Combine circuits
        combined_circuit = QuantumCircuit(self.n_qubits, name="embedding_parameterized_circuit")
        combined_circuit.compose(base_circuit, inplace=True)
        combined_circuit.compose(param_circuit, inplace=True)
        
        return combined_circuit
    
    def validate_circuit_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict:
        """
        Validate that predicted parameters can create valid quantum circuits.
        
        Args:
            parameters: Dictionary of parameter tensors
            
        Returns:
            Validation results and statistics
        """
        validation_results = {}
        
        # Check each parameter type
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
        
        # Test circuit creation with first sample
        try:
            test_circuit = self.create_parameterized_circuit(parameters, 0)
            circuit_valid = True
            circuit_depth = test_circuit.depth()
            circuit_size = test_circuit.size()
        except Exception as e:
            circuit_valid = False
            circuit_depth = 0
            circuit_size = 0
            logger.error(f"Circuit creation failed: {e}")
        
        # Overall validation
        all_finite = all(result['finite'] for result in validation_results.values())
        all_in_range = all(result['in_range'] for result in validation_results.values())
        
        validation_results['circuit_validation'] = {
            'circuit_creation_success': circuit_valid,
            'circuit_depth': circuit_depth,
            'circuit_size': circuit_size,
            'depth_within_prd_limit': circuit_depth <= 15,
            'all_parameters_finite': all_finite,
            'all_parameters_in_range': all_in_range
        }
        
        validation_results['overall'] = {
            'all_finite': all_finite,
            'all_in_range': all_in_range,
            'circuit_valid': circuit_valid,
            'prd_compliant': circuit_valid and circuit_depth <= 15,
            'valid': all_finite and all_in_range and circuit_valid
        }
        
        return validation_results
    
    def simulate_parameterized_circuit(self, 
                                     circuit: QuantumCircuit) -> Tuple[Optional[Statevector], Dict]:
        """
        Simulate a parameterized quantum circuit to get the final state.
        
        Args:
            circuit: Parameterized quantum circuit
            
        Returns:
            Tuple of (statevector, metadata)
        """
        start_time = time.time()
        
        try:
            # Use the quantum circuit simulator from core module
            simulation_result = self.quantum_circuits.simulate_circuit(circuit)
            
            execution_time = time.time() - start_time
            
            if simulation_result.success:
                metadata = {
                    'success': True,
                    'execution_time_ms': execution_time * 1000,
                    'circuit_depth': circuit.depth(),
                    'circuit_size': circuit.size(),
                    'simulation_metadata': simulation_result.metadata
                }
                return simulation_result.statevector, metadata
            else:
                metadata = {
                    'success': False,
                    'error': simulation_result.error,
                    'execution_time_ms': execution_time * 1000
                }
                return None, metadata
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Circuit simulation failed: {e}")
            metadata = {
                'success': False,
                'error': str(e),
                'execution_time_ms': execution_time * 1000
            }
            return None, metadata
    
    def batch_simulate_circuits(self, 
                              circuits: List[QuantumCircuit]) -> List[Tuple[Optional[Statevector], Dict]]:
        """
        Simulate multiple parameterized circuits efficiently.
        
        Args:
            circuits: List of quantum circuits to simulate
            
        Returns:
            List of (statevector, metadata) tuples
        """
        results = []
        
        logger.info(f"Simulating {len(circuits)} parameterized circuits")
        
        for i, circuit in enumerate(circuits):
            statevector, metadata = self.simulate_parameterized_circuit(circuit)
            
            # Add batch information
            metadata.update({
                'batch_index': i,
                'total_circuits': len(circuits)
            })
            
            results.append((statevector, metadata))
            
            # Log progress for large batches
            if len(circuits) > 10 and (i + 1) % 10 == 0:
                logger.debug(f"Simulated {i + 1}/{len(circuits)} circuits")
        
        return results
    
    def compute_circuit_fidelity(self, 
                                circuit1: QuantumCircuit, 
                                circuit2: QuantumCircuit) -> Tuple[float, Dict]:
        """
        Compute fidelity between two parameterized circuits.
        
        Args:
            circuit1: First parameterized circuit
            circuit2: Second parameterized circuit
            
        Returns:
            Tuple of (fidelity, metadata)
        """
        # Simulate both circuits
        state1, meta1 = self.simulate_parameterized_circuit(circuit1)
        state2, meta2 = self.simulate_parameterized_circuit(circuit2)
        
        if state1 is None or state2 is None:
            return 0.0, {
                'success': False,
                'error': 'Circuit simulation failed',
                'circuit1_meta': meta1,
                'circuit2_meta': meta2
            }
        
        try:
            # Compute fidelity using statevector inner product
            fidelity = float(np.abs(state1.inner(state2))**2)
            
            metadata = {
                'success': True,
                'fidelity': fidelity,
                'circuit1_meta': meta1,
                'circuit2_meta': meta2,
                'total_simulation_time_ms': meta1['execution_time_ms'] + meta2['execution_time_ms']
            }
            
            return fidelity, metadata
            
        except Exception as e:
            logger.error(f"Fidelity computation failed: {e}")
            return 0.0, {
                'success': False,
                'error': str(e),
                'circuit1_meta': meta1,
                'circuit2_meta': meta2
            }
    
    def benchmark_parameterized_circuits(self, 
                                       test_parameters: Optional[Dict[str, torch.Tensor]] = None) -> Dict:
        """
        Benchmark performance of parameterized circuit creation and simulation.
        
        Args:
            test_parameters: Optional test parameters, will generate random if None
            
        Returns:
            Performance benchmark results
        """
        if test_parameters is None:
            # Generate random test parameters
            batch_size = 5
            test_parameters = {
                'ry_params': torch.rand(batch_size, self.n_qubits * self.n_layers) * np.pi,
                'rz_params': torch.rand(batch_size, self.n_qubits * self.n_layers) * np.pi,
                'ry2_params': torch.rand(batch_size, self.n_qubits * self.n_layers) * np.pi,
                'entangling_params': torch.rand(batch_size, (self.n_qubits - 1) * self.n_layers) * np.pi
            }
        
        results = {
            'circuit_creation_times': [],
            'simulation_times': [],
            'validation_results': {},
            'performance_summary': {}
        }
        
        batch_size = test_parameters['ry_params'].shape[0]
        
        # Benchmark circuit creation
        start_time = time.time()
        circuits = self.create_batch_circuits(test_parameters)
        creation_time = time.time() - start_time
        
        results['batch_creation_time_ms'] = creation_time * 1000
        results['avg_creation_time_ms'] = (creation_time / batch_size) * 1000
        
        # Benchmark individual circuit creation
        for i in range(min(batch_size, 5)):  # Test first 5
            start_time = time.time()
            circuit = self.create_parameterized_circuit(test_parameters, i)
            creation_time = time.time() - start_time
            
            results['circuit_creation_times'].append({
                'index': i,
                'time_ms': creation_time * 1000,
                'depth': circuit.depth(),
                'size': circuit.size()
            })
        
        # Benchmark circuit simulation
        for i, circuit in enumerate(circuits[:3]):  # Test first 3 simulations
            start_time = time.time()
            statevector, metadata = self.simulate_parameterized_circuit(circuit)
            simulation_time = time.time() - start_time
            
            results['simulation_times'].append({
                'index': i,
                'time_ms': simulation_time * 1000,
                'success': metadata['success'],
                'circuit_depth': circuit.depth()
            })
        
        # Validate parameters
        validation_results = self.validate_circuit_parameters(test_parameters)
        results['validation_results'] = validation_results
        
        # Performance summary
        avg_creation_time = np.mean([r['time_ms'] for r in results['circuit_creation_times']])
        avg_simulation_time = np.mean([r['time_ms'] for r in results['simulation_times']])
        max_depth = max([r['depth'] for r in results['circuit_creation_times']])
        
        results['performance_summary'] = {
            'avg_creation_time_ms': avg_creation_time,
            'avg_simulation_time_ms': avg_simulation_time,
            'max_circuit_depth': max_depth,
            'prd_depth_compliant': max_depth <= 15,
            'all_validations_passed': validation_results['overall']['valid']
        }
        
        logger.info(f"Parameterized circuits benchmark completed: "
                   f"avg_creation={avg_creation_time:.2f}ms, "
                   f"avg_simulation={avg_simulation_time:.2f}ms")
        
        return results