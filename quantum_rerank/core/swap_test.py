"""
Quantum SWAP Test Implementation for Fidelity Computation.

This module implements the quantum SWAP test algorithm for computing fidelity 
between quantum states, providing the core similarity measurement capability
as specified in the PRD.

Based on:
- PRD Section 3.1: Core Algorithms - Quantum Fidelity via SWAP Test
- Quantum binary classifier research paper analysis
- Performance Target: <100ms per similarity computation (PRD Section 4.3)
"""

import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

logger = logging.getLogger(__name__)


@dataclass
class SWAPTestConfig:
    """Configuration for SWAP test implementation."""
    shots: int = 1024
    simulator_method: str = 'statevector'
    measurement_basis: str = 'computational'
    error_mitigation: bool = False


class QuantumSWAPTest:
    """
    Implements quantum SWAP test for fidelity computation.
    
    Based on PRD Section 3.1 and quantum algorithm research.
    The SWAP test is a fundamental quantum algorithm that allows efficient 
    computation of fidelity between two quantum states.
    """
    
    def __init__(self, n_qubits: int = 4, config: SWAPTestConfig = None):
        """
        Initialize SWAP test implementation.
        
        Args:
            n_qubits: Number of qubits per state (PRD: 2-4 qubits)
            config: SWAP test configuration
        """
        if not 2 <= n_qubits <= 4:
            raise ValueError("n_qubits must be between 2 and 4 (PRD requirement)")
        
        self.n_qubits = n_qubits
        self.config = config or SWAPTestConfig()
        
        # Total qubits: 2 states + 1 ancilla
        self.total_qubits = 2 * n_qubits + 1
        
        # Initialize quantum simulator
        self.simulator = AerSimulator(method=self.config.simulator_method)
        
        logger.info(f"SWAP test initialized: {n_qubits} qubits per state, {self.total_qubits} total")
    
    def create_swap_test_circuit(self, 
                               circuit1: QuantumCircuit, 
                               circuit2: QuantumCircuit) -> QuantumCircuit:
        """
        Create SWAP test circuit for two quantum states.
        
        The SWAP test protocol:
        1. Prepare states |ψ⟩ and |φ⟩ on separate registers
        2. Put ancilla qubit in superposition with Hadamard
        3. Apply controlled SWAP operations between state registers
        4. Apply final Hadamard to ancilla
        5. Measure ancilla qubit
        
        Theory: P(|0⟩) = 1/2 + 1/2 * |⟨ψ|φ⟩|²
        Therefore: |⟨ψ|φ⟩|² = 2 * P(|0⟩) - 1
        
        Args:
            circuit1: First quantum circuit (state |ψ⟩)
            circuit2: Second quantum circuit (state |φ⟩)
            
        Returns:
            Complete SWAP test circuit
        """
        # Validate input circuits
        if circuit1.num_qubits != self.n_qubits or circuit2.num_qubits != self.n_qubits:
            raise ValueError(f"Input circuits must have {self.n_qubits} qubits")
        
        # Create registers
        ancilla = QuantumRegister(1, 'ancilla')
        state1_reg = QuantumRegister(self.n_qubits, 'state1')
        state2_reg = QuantumRegister(self.n_qubits, 'state2')
        classical_reg = ClassicalRegister(1, 'measure')
        
        # Create SWAP test circuit
        qc = QuantumCircuit(ancilla, state1_reg, state2_reg, classical_reg, 
                           name="swap_test")
        
        # Step 1: Prepare input states
        # Compose first circuit on state1 register
        qc.compose(circuit1, qubits=state1_reg, inplace=True)
        
        # Compose second circuit on state2 register  
        qc.compose(circuit2, qubits=state2_reg, inplace=True)
        
        # Step 2: SWAP test protocol
        # Put ancilla in superposition
        qc.h(ancilla[0])
        
        # Controlled SWAP operations between corresponding qubits
        for i in range(self.n_qubits):
            qc.cswap(ancilla[0], state1_reg[i], state2_reg[i])
        
        # Final Hadamard on ancilla
        qc.h(ancilla[0])
        
        # Step 3: Measure ancilla
        qc.measure(ancilla[0], classical_reg[0])
        
        # Validate circuit depth against PRD constraints
        if qc.depth() > 15:  # PRD constraint
            logger.warning(f"SWAP test circuit depth {qc.depth()} exceeds PRD limit of 15")
        
        return qc
    
    def compute_fidelity_from_counts(self, counts: Dict[str, int]) -> float:
        """
        Compute fidelity from measurement counts.
        
        SWAP test theory: P(|0⟩) = 1/2 + 1/2 * |⟨ψ|φ⟩|²
        Therefore: |⟨ψ|φ⟩|² = 2 * P(|0⟩) - 1
        
        Args:
            counts: Measurement counts from quantum circuit
            
        Returns:
            Fidelity value between 0 and 1
        """
        total_shots = sum(counts.values())
        prob_0 = counts.get('0', 0) / total_shots
        
        # Calculate fidelity squared
        fidelity_squared = 2 * prob_0 - 1
        
        # Clamp to valid range [0, 1]
        fidelity_squared = max(0.0, min(1.0, fidelity_squared))
        
        # Return fidelity (square root of fidelity squared)
        fidelity = np.sqrt(fidelity_squared)
        
        logger.debug(f"SWAP test: P(0)={prob_0:.4f}, F²={fidelity_squared:.4f}, F={fidelity:.4f}")
        
        return float(fidelity)
    
    def compute_fidelity(self, 
                        circuit1: QuantumCircuit, 
                        circuit2: QuantumCircuit) -> Tuple[float, Dict]:
        """
        Compute quantum fidelity between two circuits using SWAP test.
        
        Args:
            circuit1: First quantum circuit
            circuit2: Second quantum circuit
            
        Returns:
            Tuple of (fidelity, metadata)
        """
        start_time = time.time()
        
        try:
            # Create SWAP test circuit
            swap_circuit = self.create_swap_test_circuit(circuit1, circuit2)
            
            # Execute circuit
            job = self.simulator.run(swap_circuit, shots=self.config.shots)
            result = job.result()
            counts = result.get_counts(swap_circuit)
            
            # Compute fidelity
            fidelity = self.compute_fidelity_from_counts(counts)
            
            # Collect metadata
            execution_time = time.time() - start_time
            metadata = {
                'execution_time_ms': execution_time * 1000,
                'shots': self.config.shots,
                'circuit_depth': swap_circuit.depth(),
                'circuit_size': swap_circuit.size(),
                'measurement_counts': counts,
                'success': True
            }
            
            # Check PRD performance target
            if execution_time * 1000 > 100:  # PRD: <100ms per similarity
                logger.warning(f"Fidelity computation took {execution_time*1000:.2f}ms, exceeds PRD target of 100ms")
            
            return fidelity, metadata
            
        except Exception as e:
            logger.error(f"Fidelity computation failed: {e}")
            metadata = {
                'success': False,
                'error': str(e),
                'execution_time_ms': (time.time() - start_time) * 1000
            }
            return 0.0, metadata

    def compute_fidelity_statevector(self, 
                                    circuit1: QuantumCircuit, 
                                    circuit2: QuantumCircuit) -> Tuple[float, Dict]:
        """
        Compute fidelity using statevector method (faster for simulation).
        
        Alternative method for performance comparison and validation.
        Uses direct quantum information formula: F = |⟨ψ|φ⟩|²
        """
        start_time = time.time()
        
        try:
            # Get statevectors directly
            statevector1 = Statevector.from_instruction(circuit1)
            statevector2 = Statevector.from_instruction(circuit2)
            
            # Compute fidelity using quantum information formula
            fidelity = float(np.abs(statevector1.inner(statevector2))**2)
            
            execution_time = time.time() - start_time
            metadata = {
                'method': 'statevector',
                'execution_time_ms': execution_time * 1000,
                'success': True
            }
            
            return fidelity, metadata
            
        except Exception as e:
            logger.error(f"Statevector fidelity computation failed: {e}")
            metadata = {
                'method': 'statevector',
                'success': False,
                'error': str(e),
                'execution_time_ms': (time.time() - start_time) * 1000
            }
            return 0.0, metadata

    def batch_compute_fidelity(self, 
                              query_circuit: QuantumCircuit,
                              candidate_circuits: List[QuantumCircuit]) -> List[Tuple[float, Dict]]:
        """
        Compute fidelity between query and multiple candidates efficiently.
        
        Supports PRD batch processing requirements (50-100 candidates).
        """
        results = []
        
        logger.info(f"Computing fidelity for {len(candidate_circuits)} candidates")
        
        for i, candidate_circuit in enumerate(candidate_circuits):
            fidelity, metadata = self.compute_fidelity(query_circuit, candidate_circuit)
            
            # Add batch information to metadata
            metadata.update({
                'batch_index': i,
                'total_candidates': len(candidate_circuits)
            })
            
            results.append((fidelity, metadata))
            
            # Log progress for large batches
            if len(candidate_circuits) > 10 and (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{len(candidate_circuits)} candidates")
        
        return results

    def validate_swap_test_implementation(self) -> Dict:
        """
        Validate SWAP test implementation against known results.
        
        Tests with known quantum states to ensure correctness.
        """
        from ..quantum_circuits import BasicQuantumCircuits
        
        qc_handler = BasicQuantumCircuits(self.n_qubits)
        
        test_results = {}
        
        # Test 1: Identical states should have fidelity = 1
        identical_circuit = qc_handler.create_superposition_circuit()
        fidelity, metadata = self.compute_fidelity(identical_circuit, identical_circuit)
        
        test_results['identical_states'] = {
            'fidelity': fidelity,
            'expected': 1.0,
            'error': abs(fidelity - 1.0),
            'pass': abs(fidelity - 1.0) < 0.1  # Allow some measurement error
        }
        
        # Test 2: Orthogonal states should have fidelity = 0
        zero_state = qc_handler.create_empty_circuit()  # |00...0⟩
        one_state = QuantumCircuit(self.n_qubits)
        one_state.x(0)  # |10...0⟩
        
        fidelity, metadata = self.compute_fidelity(zero_state, one_state)
        
        test_results['orthogonal_states'] = {
            'fidelity': fidelity,
            'expected': 0.0,
            'error': abs(fidelity - 0.0),
            'pass': abs(fidelity - 0.0) < 0.1  # Allow some measurement error
        }
        
        # Test 3: Compare SWAP test vs statevector method
        test_circuit1 = qc_handler.create_superposition_circuit()
        test_circuit2 = qc_handler.create_entanglement_circuit()
        
        fidelity_swap, _ = self.compute_fidelity(test_circuit1, test_circuit2)
        fidelity_sv, _ = self.compute_fidelity_statevector(test_circuit1, test_circuit2)
        
        test_results['method_comparison'] = {
            'swap_test': fidelity_swap,
            'statevector': fidelity_sv,
            'difference': abs(fidelity_swap - fidelity_sv),
            'pass': abs(fidelity_swap - fidelity_sv) < 0.1
        }
        
        # Overall validation
        all_tests_pass = all(result.get('pass', False) for result in test_results.values())
        test_results['overall_pass'] = all_tests_pass
        
        logger.info(f"SWAP test validation: {'PASS' if all_tests_pass else 'FAIL'}")
        
        return test_results

    def benchmark_swap_test_performance(self) -> Dict:
        """
        Benchmark SWAP test performance against PRD targets.
        
        Returns performance metrics for optimization.
        """
        from ..quantum_circuits import BasicQuantumCircuits
        
        qc_handler = BasicQuantumCircuits(self.n_qubits)
        
        # Create test circuits
        test_circuits = [
            qc_handler.create_empty_circuit(),
            qc_handler.create_superposition_circuit(),
            qc_handler.create_entanglement_circuit()
        ]
        
        results = {
            'single_fidelity_times': [],
            'batch_processing_time': 0,
            'memory_usage': 0
        }
        
        # Benchmark single fidelity computations
        for i, circuit1 in enumerate(test_circuits):
            for j, circuit2 in enumerate(test_circuits):
                start_time = time.time()
                fidelity, metadata = self.compute_fidelity(circuit1, circuit2)
                execution_time = time.time() - start_time
                
                results['single_fidelity_times'].append({
                    'circuit_pair': (i, j),
                    'time_ms': execution_time * 1000,
                    'fidelity': fidelity,
                    'meets_target': execution_time * 1000 < 100  # PRD target
                })
        
        # Benchmark batch processing
        query_circuit = test_circuits[0]
        start_time = time.time()
        batch_results = self.batch_compute_fidelity(query_circuit, test_circuits)
        batch_time = time.time() - start_time
        
        results['batch_processing_time'] = batch_time * 1000
        results['batch_per_item_ms'] = (batch_time / len(test_circuits)) * 1000
        
        # Performance summary
        avg_single_time = np.mean([r['time_ms'] for r in results['single_fidelity_times']])
        max_single_time = np.max([r['time_ms'] for r in results['single_fidelity_times']])
        
        results['performance_summary'] = {
            'avg_single_fidelity_ms': avg_single_time,
            'max_single_fidelity_ms': max_single_time,
            'meets_prd_target': max_single_time < 100,
            'batch_efficiency': results['batch_per_item_ms'] < avg_single_time
        }
        
        logger.info(f"SWAP test performance: avg={avg_single_time:.2f}ms, max={max_single_time:.2f}ms")
        
        return results