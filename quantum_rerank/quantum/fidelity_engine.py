"""
Optimized Quantum Fidelity Computation Engine.

This module implements optimized quantum fidelity computation using SWAP test
and direct fidelity methods for accurate similarity measurement between quantum
states representing embeddings.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector, state_fidelity

from ..utils import get_logger, QuantumRerankException
from ..config import Configurable, QuantumRerankConfigSchema


@dataclass
class FidelityPerformanceTargets:
    """Performance targets for fidelity computation."""
    single_computation_ms: float = 85.0    # PRD: <100ms target
    batch_50_docs_ms: float = 400.0        # Efficient batch processing
    batch_100_docs_ms: float = 750.0       # Maximum batch size
    fidelity_precision: float = 0.001      # Sufficient for ranking
    ranking_correlation: float = 0.95      # High ranking quality
    noise_tolerance: float = 0.02          # Robust to quantum noise
    memory_per_computation_mb: float = 10.0 # Efficient memory usage
    quantum_gate_count_max: int = 50       # Hardware-efficient circuits


@dataclass
class FidelityResult:
    """Result of fidelity computation."""
    fidelity: float
    computation_time_ms: float
    method_used: str
    gate_count: int
    cache_hit: bool = False
    error: Optional[str] = None


class FidelityMethod(ABC):
    """Abstract base class for fidelity computation methods."""
    
    @abstractmethod
    def compute_fidelity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        **kwargs
    ) -> FidelityResult:
        """Compute fidelity between two embeddings."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get method name."""
        pass
    
    @abstractmethod
    def estimate_performance(self, embedding_size: int) -> Dict[str, float]:
        """Estimate performance for given embedding size."""
        pass


class OptimizedSWAPTest(FidelityMethod):
    """Hardware-efficient SWAP test for quantum fidelity computation."""
    
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.ancilla_qubit = n_qubits
        self.total_qubits = 2 * n_qubits + 1
        self.logger = get_logger(__name__)
        
        # Initialize quantum backend
        self.backend = AerSimulator(method='statevector')
        
        # Performance tracking
        self.gate_count_cache = {}
    
    def get_method_name(self) -> str:
        return "optimized_swap_test"
    
    def estimate_performance(self, embedding_size: int) -> Dict[str, float]:
        """Estimate performance metrics."""
        # Empirical estimates based on circuit complexity
        estimated_gates = 3 + 2 * self.n_qubits  # H + CSWAPs + H
        estimated_time_ms = 50 + embedding_size * 0.1  # Base time + encoding overhead
        
        return {
            "estimated_time_ms": estimated_time_ms,
            "estimated_gates": estimated_gates,
            "memory_mb": self.n_qubits * 2  # Rough estimate
        }
    
    def create_fidelity_circuit(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> QuantumCircuit:
        """Create optimized SWAP test circuit."""
        # Quantum and classical registers
        qreg = QuantumRegister(self.total_qubits, 'q')
        creg = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        # Normalize embeddings
        embedding1 = self._normalize_embedding(embedding1)
        embedding2 = self._normalize_embedding(embedding2)
        
        # State preparation with amplitude encoding
        self._encode_embedding_optimized(qc, embedding1, list(range(self.n_qubits)))
        self._encode_embedding_optimized(qc, embedding2, list(range(self.n_qubits, 2 * self.n_qubits)))
        
        # Hadamard on ancilla qubit
        qc.h(self.ancilla_qubit)
        
        # Controlled SWAP operations
        for i in range(self.n_qubits):
            qc.cswap(self.ancilla_qubit, i, i + self.n_qubits)
        
        # Final Hadamard and measurement
        qc.h(self.ancilla_qubit)
        qc.measure(self.ancilla_qubit, 0)
        
        return qc
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for quantum state preparation."""
        # Truncate to fit quantum register
        max_size = 2 ** self.n_qubits
        if len(embedding) > max_size:
            embedding = embedding[:max_size]
        elif len(embedding) < max_size:
            # Pad with zeros
            embedding = np.pad(embedding, (0, max_size - len(embedding)))
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _encode_embedding_optimized(
        self, 
        circuit: QuantumCircuit, 
        embedding: np.ndarray, 
        qubits: List[int]
    ) -> None:
        """Optimized amplitude encoding for embeddings."""
        try:
            # Use Qiskit's state preparation with optimization
            state_prep = StatePreparation(embedding, inverse=False, label='embedding')
            circuit.append(state_prep, qubits)
        except Exception as e:
            self.logger.warning(f"Failed optimized encoding, using manual method: {e}")
            self._manual_amplitude_encoding(circuit, embedding, qubits)
    
    def _manual_amplitude_encoding(
        self, 
        circuit: QuantumCircuit, 
        embedding: np.ndarray, 
        qubits: List[int]
    ) -> None:
        """Manual amplitude encoding as fallback."""
        # Simple rotation-based encoding for small embeddings
        for i, qubit in enumerate(qubits):
            if i < len(embedding):
                # Use rotation angles based on embedding values
                theta = 2 * np.arcsin(min(abs(embedding[i]), 1.0))
                circuit.ry(theta, qubit)
    
    def compute_fidelity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        **kwargs
    ) -> FidelityResult:
        """Compute fidelity using optimized SWAP test."""
        start_time = time.time()
        
        try:
            # Create and execute circuit
            circuit = self.create_fidelity_circuit(embedding1, embedding2)
            
            # Count gates for performance tracking
            gate_count = len(circuit.data)
            
            # Execute circuit
            job = execute(circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)
            
            # Calculate fidelity from measurement statistics
            p0 = counts.get('0', 0) / self.shots
            fidelity = max(0.0, 2 * p0 - 1)  # Ensure non-negative
            
            computation_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return FidelityResult(
                fidelity=fidelity,
                computation_time_ms=computation_time,
                method_used=self.get_method_name(),
                gate_count=gate_count,
                cache_hit=False
            )
        
        except Exception as e:
            computation_time = (time.time() - start_time) * 1000
            self.logger.error(f"SWAP test fidelity computation failed: {e}")
            
            return FidelityResult(
                fidelity=0.0,
                computation_time_ms=computation_time,
                method_used=self.get_method_name(),
                gate_count=0,
                error=str(e)
            )


class DirectStatevectorFidelity(FidelityMethod):
    """Direct statevector fidelity computation for simulators."""
    
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.logger = get_logger(__name__)
    
    def get_method_name(self) -> str:
        return "direct_statevector"
    
    def estimate_performance(self, embedding_size: int) -> Dict[str, float]:
        """Estimate performance for direct computation."""
        # Direct computation is much faster but requires exponential memory
        return {
            "estimated_time_ms": 5.0,  # Very fast
            "estimated_gates": 0,      # No quantum gates needed
            "memory_mb": 2 ** self.n_qubits * 0.032  # Statevector memory
        }
    
    def compute_fidelity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        **kwargs
    ) -> FidelityResult:
        """Compute fidelity directly from statevectors."""
        start_time = time.time()
        
        try:
            # Normalize embeddings
            state1 = self._prepare_statevector(embedding1)
            state2 = self._prepare_statevector(embedding2)
            
            # Compute fidelity directly
            fidelity = abs(np.dot(state1.conj(), state2)) ** 2
            
            computation_time = (time.time() - start_time) * 1000
            
            return FidelityResult(
                fidelity=float(fidelity),
                computation_time_ms=computation_time,
                method_used=self.get_method_name(),
                gate_count=0,
                cache_hit=False
            )
        
        except Exception as e:
            computation_time = (time.time() - start_time) * 1000
            self.logger.error(f"Direct fidelity computation failed: {e}")
            
            return FidelityResult(
                fidelity=0.0,
                computation_time_ms=computation_time,
                method_used=self.get_method_name(),
                gate_count=0,
                error=str(e)
            )
    
    def _prepare_statevector(self, embedding: np.ndarray) -> np.ndarray:
        """Prepare quantum statevector from embedding."""
        # Truncate/pad to fit quantum register
        max_size = 2 ** self.n_qubits
        if len(embedding) > max_size:
            state = embedding[:max_size]
        elif len(embedding) < max_size:
            state = np.pad(embedding, (0, max_size - len(embedding)))
        else:
            state = embedding.copy()
        
        # Normalize
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        
        return state


class ApproximateClassicalFidelity(FidelityMethod):
    """Classical approximation for fast fidelity computation."""
    
    def get_method_name(self) -> str:
        return "approximate_classical"
    
    def estimate_performance(self, embedding_size: int) -> Dict[str, float]:
        """Estimate performance for classical approximation."""
        return {
            "estimated_time_ms": 1.0,  # Very fast
            "estimated_gates": 0,      # Classical only
            "memory_mb": embedding_size * 0.008  # Just vector operations
        }
    
    def compute_fidelity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        **kwargs
    ) -> FidelityResult:
        """Compute approximate fidelity using classical methods."""
        start_time = time.time()
        
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                fidelity = 0.0
            else:
                # Use cosine similarity as fidelity approximation
                cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
                fidelity = (cosine_sim + 1) / 2  # Map [-1,1] to [0,1]
            
            computation_time = (time.time() - start_time) * 1000
            
            return FidelityResult(
                fidelity=float(fidelity),
                computation_time_ms=computation_time,
                method_used=self.get_method_name(),
                gate_count=0,
                cache_hit=False
            )
        
        except Exception as e:
            computation_time = (time.time() - start_time) * 1000
            
            return FidelityResult(
                fidelity=0.0,
                computation_time_ms=computation_time,
                method_used=self.get_method_name(),
                gate_count=0,
                error=str(e)
            )


class FidelityCache:
    """Intelligent caching for fidelity computations."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, FidelityResult] = {}
        self.access_count: Dict[str, int] = {}
        self.lock = threading.RLock()
        self.logger = get_logger(__name__)
    
    def _compute_hash(self, embedding1: np.ndarray, embedding2: np.ndarray) -> str:
        """Compute hash key for embedding pair."""
        # Create a deterministic hash from both embeddings
        combined = np.concatenate([embedding1.flatten(), embedding2.flatten()])
        return hashlib.md5(combined.tobytes()).hexdigest()
    
    def get(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> Optional[FidelityResult]:
        """Get cached fidelity result."""
        key = self._compute_hash(embedding1, embedding2)
        
        with self.lock:
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                result = self.cache[key]
                # Mark as cache hit
                result.cache_hit = True
                return result
        
        return None
    
    def put(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray, 
        result: FidelityResult
    ) -> None:
        """Cache fidelity result."""
        key = self._compute_hash(embedding1, embedding2)
        
        with self.lock:
            # Manage cache size
            if len(self.cache) >= self.max_size:
                self._evict_least_used()
            
            self.cache[key] = result
            self.access_count[key] = 1
    
    def _evict_least_used(self) -> None:
        """Evict least frequently used entries."""
        if not self.cache:
            return
        
        # Find least used key
        min_access = min(self.access_count.values())
        keys_to_remove = [k for k, v in self.access_count.items() if v == min_access]
        
        # Remove one of the least used keys
        key_to_remove = keys_to_remove[0]
        del self.cache[key_to_remove]
        del self.access_count[key_to_remove]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_access = sum(self.access_count.values())
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "total_accesses": total_access,
                "hit_rate": len(self.cache) / max(total_access, 1),
                "memory_usage_estimate_mb": len(self.cache) * 0.1  # Rough estimate
            }


class QuantumFidelityEngine(Configurable):
    """
    Main quantum fidelity computation engine with method selection and optimization.
    
    This engine automatically selects the optimal fidelity computation method
    based on requirements and performance targets.
    """
    
    def __init__(
        self, 
        n_qubits: int = 4,
        enable_caching: bool = True,
        performance_targets: Optional[FidelityPerformanceTargets] = None
    ):
        self.n_qubits = n_qubits
        self.enable_caching = enable_caching
        self.performance_targets = performance_targets or FidelityPerformanceTargets()
        self.logger = get_logger(__name__)
        
        # Initialize computation methods
        self.methods: Dict[str, FidelityMethod] = {
            "swap_test": OptimizedSWAPTest(n_qubits),
            "direct_statevector": DirectStatevectorFidelity(n_qubits),
            "approximate_classical": ApproximateClassicalFidelity()
        }
        
        # Initialize cache
        self.cache = FidelityCache() if enable_caching else None
        
        # Performance monitoring
        self.computation_history: List[FidelityResult] = []
        
        self.logger.info(f"Initialized QuantumFidelityEngine with {len(self.methods)} methods")
    
    def select_fidelity_method(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        accuracy_requirement: float = 0.95,
        time_limit_ms: Optional[float] = None
    ) -> str:
        """
        Choose optimal fidelity computation method based on requirements.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            accuracy_requirement: Required accuracy level
            time_limit_ms: Time limit for computation
            
        Returns:
            Method name to use
        """
        embedding_size = max(len(embedding1), len(embedding2))
        time_limit = time_limit_ms or self.performance_targets.single_computation_ms
        
        # Method selection logic based on PRD requirements
        if embedding_size <= 16 and accuracy_requirement >= 0.99:
            return "direct_statevector"  # Exact and fast for small states
        elif accuracy_requirement >= 0.95 and time_limit >= 50:
            return "swap_test"           # High accuracy quantum method
        else:
            return "approximate_classical"  # Fast approximation
    
    def compute_fidelity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        method: Optional[str] = None,
        accuracy_requirement: float = 0.95,
        use_cache: bool = True
    ) -> FidelityResult:
        """
        Compute fidelity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Specific method to use (auto-select if None)
            accuracy_requirement: Required accuracy level
            use_cache: Whether to use caching
            
        Returns:
            FidelityResult with computation details
        """
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get(embedding1, embedding2)
            if cached_result:
                self.logger.debug("Cache hit for fidelity computation")
                return cached_result
        
        # Select method if not specified
        if method is None:
            method = self.select_fidelity_method(
                embedding1, embedding2, accuracy_requirement
            )
        
        # Validate method
        if method not in self.methods:
            self.logger.warning(f"Unknown method '{method}', using approximate_classical")
            method = "approximate_classical"
        
        # Compute fidelity
        result = self.methods[method].compute_fidelity(embedding1, embedding2)
        
        # Cache result if successful
        if use_cache and self.cache and result.error is None:
            self.cache.put(embedding1, embedding2, result)
        
        # Update performance history
        self.computation_history.append(result)
        if len(self.computation_history) > 1000:  # Keep last 1000 computations
            self.computation_history = self.computation_history[-1000:]
        
        return result
    
    def compute_batch_fidelity(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray],
        method: Optional[str] = None,
        max_workers: int = 4
    ) -> List[FidelityResult]:
        """
        Compute fidelity for query against multiple candidates efficiently.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            method: Fidelity computation method
            max_workers: Maximum parallel workers
            
        Returns:
            List of fidelity results in same order as candidates
        """
        start_time = time.time()
        results = [None] * len(candidate_embeddings)
        
        # Use parallel computation for large batches
        if len(candidate_embeddings) > 10 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all computations
                future_to_index = {
                    executor.submit(
                        self.compute_fidelity, 
                        query_embedding, 
                        candidate, 
                        method
                    ): i
                    for i, candidate in enumerate(candidate_embeddings)
                }
                
                # Collect results
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        self.logger.error(f"Batch computation failed for index {index}: {e}")
                        results[index] = FidelityResult(
                            fidelity=0.0,
                            computation_time_ms=0.0,
                            method_used="error",
                            gate_count=0,
                            error=str(e)
                        )
        else:
            # Sequential computation for small batches
            for i, candidate in enumerate(candidate_embeddings):
                results[i] = self.compute_fidelity(query_embedding, candidate, method)
        
        total_time = (time.time() - start_time) * 1000
        self.logger.info(
            f"Batch fidelity computation completed: {len(candidate_embeddings)} "
            f"candidates in {total_time:.2f}ms"
        )
        
        return results
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.computation_history:
            return {"message": "No computations performed yet"}
        
        recent_computations = self.computation_history[-100:]  # Last 100
        
        # Compute statistics by method
        method_stats = {}
        for method_name in self.methods.keys():
            method_computations = [
                c for c in recent_computations 
                if c.method_used == method_name and c.error is None
            ]
            
            if method_computations:
                times = [c.computation_time_ms for c in method_computations]
                fidelities = [c.fidelity for c in method_computations]
                
                method_stats[method_name] = {
                    "count": len(method_computations),
                    "avg_time_ms": np.mean(times),
                    "max_time_ms": np.max(times),
                    "avg_fidelity": np.mean(fidelities),
                    "meets_target": np.mean(times) < self.performance_targets.single_computation_ms
                }
        
        # Overall statistics
        all_times = [c.computation_time_ms for c in recent_computations if c.error is None]
        cache_hits = sum(1 for c in recent_computations if c.cache_hit)
        
        stats = {
            "total_computations": len(self.computation_history),
            "recent_computations": len(recent_computations),
            "method_statistics": method_stats,
            "overall_avg_time_ms": np.mean(all_times) if all_times else 0,
            "cache_hit_rate": cache_hits / len(recent_computations) if recent_computations else 0,
            "performance_target_met": (
                np.mean(all_times) < self.performance_targets.single_computation_ms 
                if all_times else False
            )
        }
        
        # Add cache statistics if available
        if self.cache:
            stats["cache_statistics"] = self.cache.get_statistics()
        
        return stats
    
    def configure(self, config: QuantumRerankConfigSchema) -> None:
        """Configure engine with new configuration."""
        if hasattr(config, 'quantum'):
            quantum_config = config.quantum
            if quantum_config.n_qubits != self.n_qubits:
                self.logger.info(f"Updating n_qubits from {self.n_qubits} to {quantum_config.n_qubits}")
                self.n_qubits = quantum_config.n_qubits
                # Reinitialize methods with new qubit count
                self._reinitialize_methods()
    
    def validate_config(self, config: QuantumRerankConfigSchema) -> bool:
        """Validate configuration for this component."""
        if hasattr(config, 'quantum'):
            quantum_config = config.quantum
            # Check if quantum configuration is reasonable
            if quantum_config.n_qubits < 2 or quantum_config.n_qubits > 8:
                return False
        return True
    
    def get_config_requirements(self) -> List[str]:
        """Get list of required configuration sections."""
        return ["quantum", "performance"]
    
    def _reinitialize_methods(self) -> None:
        """Reinitialize computation methods with new parameters."""
        self.methods = {
            "swap_test": OptimizedSWAPTest(self.n_qubits),
            "direct_statevector": DirectStatevectorFidelity(self.n_qubits),
            "approximate_classical": ApproximateClassicalFidelity()
        }


__all__ = [
    "FidelityPerformanceTargets",
    "FidelityResult",
    "FidelityMethod",
    "OptimizedSWAPTest",
    "DirectStatevectorFidelity",
    "ApproximateClassicalFidelity",
    "FidelityCache",
    "QuantumFidelityEngine"
]