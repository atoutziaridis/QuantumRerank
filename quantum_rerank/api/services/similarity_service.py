"""
Similarity service for QuantumRerank API.

This service provides async wrappers and business logic for quantum similarity
computations, integrating with the core quantum engine while providing
API-optimized interfaces.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from ...core.rag_reranker import QuantumRAGReranker
from ...utils.logging_config import get_logger
from ...monitoring.performance_monitor import PerformanceMonitor
from ..models import SimilarityMethod

logger = get_logger(__name__)


class SimilarityService:
    """
    Service layer for quantum similarity computations.
    
    Provides async interfaces and business logic for the API endpoints,
    wrapping the synchronous quantum engine with async execution.
    """
    
    def __init__(
        self,
        quantum_reranker: QuantumRAGReranker,
        performance_monitor: PerformanceMonitor,
        max_workers: int = 4
    ):
        """
        Initialize similarity service.
        
        Args:
            quantum_reranker: Core quantum reranker instance
            performance_monitor: Performance monitoring instance
            max_workers: Maximum threads for async execution
        """
        self.quantum_reranker = quantum_reranker
        self.performance_monitor = performance_monitor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logger
        
        # Service metrics
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
    
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        method: SimilarityMethod = SimilarityMethod.HYBRID,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute similarity between two texts asynchronously.
        
        Args:
            text1: First text for comparison
            text2: Second text for comparison
            method: Similarity computation method
            request_id: Optional request identifier
            
        Returns:
            Dictionary with similarity score and metadata
        """
        start_time = time.perf_counter()
        request_id = request_id or str(int(time.time() * 1000000))
        
        self.logger.info(
            "Computing similarity",
            extra={
                "request_id": request_id,
                "method": method.value,
                "text1_length": len(text1),
                "text2_length": len(text2)
            }
        )
        
        try:
            # Execute similarity computation in thread pool
            similarity_score = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._compute_similarity_sync,
                text1,
                text2,
                method.value
            )
            
            # Calculate timing
            processing_time = time.perf_counter() - start_time
            
            # Get computation details from quantum engine
            engine_stats = self.quantum_reranker.get_performance_stats()
            
            # Prepare response with metadata
            result = {
                "similarity_score": similarity_score,
                "method_used": method.value,
                "computation_time_ms": processing_time * 1000,
                "computation_details": self._extract_computation_details(method.value, engine_stats),
                "request_metadata": {
                    "request_id": request_id,
                    "text1_length": len(text1),
                    "text2_length": len(text2),
                    "timestamp": time.time()
                }
            }
            
            # Update service metrics
            self._update_metrics(processing_time, success=True)
            
            self.logger.info(
                "Similarity computation completed",
                extra={
                    "request_id": request_id,
                    "similarity_score": similarity_score,
                    "processing_time_ms": processing_time * 1000
                }
            )
            
            return result
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self._update_metrics(processing_time, success=False)
            
            self.logger.error(
                "Similarity computation failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "processing_time_ms": processing_time * 1000
                }
            )
            raise e
    
    async def rerank_documents(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 10,
        method: SimilarityMethod = SimilarityMethod.HYBRID,
        user_context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rerank documents using quantum similarity asynchronously.
        
        Args:
            query: Query text
            candidates: List of candidate documents
            top_k: Number of top results to return
            method: Similarity computation method
            user_context: Optional user context for personalization
            request_id: Optional request identifier
            
        Returns:
            Dictionary with ranked results and metadata
        """
        start_time = time.perf_counter()
        request_id = request_id or str(int(time.time() * 1000000))
        
        self.logger.info(
            "Starting document reranking",
            extra={
                "request_id": request_id,
                "method": method.value,
                "query_length": len(query),
                "candidate_count": len(candidates),
                "top_k": top_k
            }
        )
        
        try:
            # Execute reranking in thread pool
            reranked_results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._rerank_documents_sync,
                query,
                candidates,
                top_k,
                method.value,
                user_context
            )
            
            # Calculate timing
            processing_time = time.perf_counter() - start_time
            
            # Get engine statistics
            engine_stats = self.quantum_reranker.get_performance_stats()
            
            # Format results with ranks and metadata
            formatted_results = []
            for rank, (text, score, metadata) in enumerate(reranked_results, 1):
                formatted_results.append({
                    "text": text,
                    "similarity_score": score,
                    "rank": rank,
                    "metadata": metadata or {}
                })
            
            # Prepare response
            result = {
                "results": formatted_results,
                "query_metadata": {
                    "total_candidates": len(candidates),
                    "returned_results": len(formatted_results),
                    "method": method.value,
                    "embedding_model": getattr(self.quantum_reranker, 'model_name', 'unknown'),
                    "quantum_backend": getattr(self.quantum_reranker.similarity_engine, 'quantum_backend', 'unknown') if hasattr(self.quantum_reranker, 'similarity_engine') else 'unknown'
                },
                "computation_time_ms": processing_time * 1000,
                "method_used": method.value,
                "request_metadata": {
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "user_context_provided": user_context is not None
                }
            }
            
            # Update service metrics
            self._update_metrics(processing_time, success=True)
            
            self.logger.info(
                "Document reranking completed",
                extra={
                    "request_id": request_id,
                    "results_count": len(formatted_results),
                    "processing_time_ms": processing_time * 1000
                }
            )
            
            return result
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self._update_metrics(processing_time, success=False)
            
            self.logger.error(
                "Document reranking failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "processing_time_ms": processing_time * 1000
                }
            )
            raise e
    
    async def batch_similarity(
        self,
        query: str,
        candidates: List[str],
        method: SimilarityMethod = SimilarityMethod.HYBRID,
        threshold: Optional[float] = None,
        return_all: bool = False,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute batch similarities efficiently.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            method: Similarity computation method
            threshold: Minimum similarity threshold
            return_all: Whether to return all results or only above threshold
            request_id: Optional request identifier
            
        Returns:
            Dictionary with batch similarity results
        """
        start_time = time.perf_counter()
        request_id = request_id or str(int(time.time() * 1000000))
        
        self.logger.info(
            "Starting batch similarity computation",
            extra={
                "request_id": request_id,
                "method": method.value,
                "query_length": len(query),
                "candidate_count": len(candidates),
                "threshold": threshold
            }
        )
        
        try:
            # Execute batch computation in thread pool
            similarity_results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._batch_similarity_sync,
                query,
                candidates,
                method.value,
                threshold,
                return_all
            )
            
            # Calculate timing
            processing_time = time.perf_counter() - start_time
            
            # Prepare response
            result = {
                "results": similarity_results,
                "total_processed": len(candidates),
                "results_returned": len(similarity_results),
                "computation_time_ms": processing_time * 1000,
                "method_used": method.value,
                "request_metadata": {
                    "request_id": request_id,
                    "threshold_applied": threshold,
                    "return_all": return_all,
                    "timestamp": time.time()
                }
            }
            
            # Update service metrics
            self._update_metrics(processing_time, success=True)
            
            self.logger.info(
                "Batch similarity computation completed",
                extra={
                    "request_id": request_id,
                    "results_count": len(similarity_results),
                    "processing_time_ms": processing_time * 1000
                }
            )
            
            return result
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self._update_metrics(processing_time, success=False)
            
            self.logger.error(
                "Batch similarity computation failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "processing_time_ms": processing_time * 1000
                }
            )
            raise e
    
    def _compute_similarity_sync(self, text1: str, text2: str, method: str) -> float:
        """Synchronous similarity computation."""
        return self.quantum_reranker.compute_similarity(text1, text2, method)
    
    def _rerank_documents_sync(
        self,
        query: str,
        candidates: List[str],
        top_k: int,
        method: str,
        user_context: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Synchronous document reranking."""
        results = self.quantum_reranker.rerank_candidates(
            query,
            candidates,
            top_k=top_k,
            method=method
        )
        
        # Format results as tuples with metadata
        formatted_results = []
        for result in results:
            text = result.get("text", "")
            score = result.get("similarity_score", 0.0)
            metadata = result.get("metadata", {})
            formatted_results.append((text, score, metadata))
        
        return formatted_results
    
    def _batch_similarity_sync(
        self,
        query: str,
        candidates: List[str],
        method: str,
        threshold: Optional[float],
        return_all: bool
    ) -> List[Dict[str, Any]]:
        """Synchronous batch similarity computation."""
        results = []
        
        for i, candidate in enumerate(candidates):
            try:
                score = self.quantum_reranker.compute_similarity(query, candidate, method)
                
                # Apply threshold filter
                if threshold is None or score >= threshold or return_all:
                    results.append({
                        "index": i,
                        "text": candidate,
                        "similarity_score": score,
                        "above_threshold": threshold is None or score >= threshold
                    })
                    
            except Exception as e:
                # Log error but continue processing
                self.logger.warning(
                    f"Failed to compute similarity for candidate {i}: {e}"
                )
                if return_all:
                    results.append({
                        "index": i,
                        "text": candidate,
                        "similarity_score": 0.0,
                        "error": str(e),
                        "above_threshold": False
                    })
        
        return results
    
    def _extract_computation_details(self, method: str, engine_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract computation details based on method used."""
        details = {
            "method": method,
            "computation_backend": "quantum_enhanced"
        }
        
        # Add method-specific details
        if method == "quantum":
            details.update({
                "quantum_fidelity_used": True,
                "circuit_depth": engine_stats.get("avg_circuit_depth", "unknown"),
                "num_qubits": engine_stats.get("num_qubits", "unknown"),
                "backend": engine_stats.get("quantum_backend", "unknown")
            })
        elif method == "classical":
            details.update({
                "cosine_similarity_used": True,
                "embedding_dimension": engine_stats.get("embedding_dim", "unknown")
            })
        elif method == "hybrid":
            details.update({
                "hybrid_weighting": True,
                "classical_weight": 0.5,  # From configuration
                "quantum_weight": 0.5,
                "quantum_fidelity_used": True,
                "cosine_similarity_used": True
            })
        
        return details
    
    def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update service metrics."""
        self.request_count += 1
        self.total_processing_time += processing_time
        
        if not success:
            self.error_count += 1
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get current service metrics.
        
        Returns:
            Dictionary with service performance metrics
        """
        if self.request_count == 0:
            return {
                "request_count": 0,
                "average_processing_time_ms": 0.0,
                "error_rate": 0.0,
                "total_processing_time_ms": 0.0
            }
        
        avg_processing_time = (self.total_processing_time / self.request_count) * 1000
        error_rate = (self.error_count / self.request_count) * 100
        
        return {
            "request_count": self.request_count,
            "average_processing_time_ms": avg_processing_time,
            "error_rate": error_rate,
            "total_processing_time_ms": self.total_processing_time * 1000,
            "successful_requests": self.request_count - self.error_count,
            "failed_requests": self.error_count
        }
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        self.executor.shutdown(wait=True)
        self.logger.info("SimilarityService cleanup completed")