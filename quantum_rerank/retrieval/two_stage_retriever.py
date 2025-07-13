"""
Two-stage retrieval pipeline combining FAISS and quantum reranking.

This module implements the complete retrieval pipeline as specified in
PRD Section 5.2, with FAISS for initial retrieval and quantum reranking
for final results.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from .faiss_store import QuantumFAISSStore, FAISSConfig, SearchResult
from .document_store import DocumentStore, Document
from ..core.rag_reranker import QuantumRAGReranker
from ..core.quantum_similarity_engine import SimilarityEngineConfig, SimilarityMethod
from ..core.embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Configuration for two-stage retriever."""
    # FAISS configuration
    faiss_config: FAISSConfig = field(default_factory=FAISSConfig)
    
    # Retrieval parameters
    initial_k: int = 100  # Number of candidates from FAISS
    final_k: int = 10    # Number of results after reranking
    
    # Quantum reranking configuration
    reranking_method: str = "hybrid"  # classical, quantum, or hybrid
    similarity_engine_config: Optional[SimilarityEngineConfig] = None
    
    # Performance settings
    enable_caching: bool = True
    batch_size: int = 32
    
    # Fallback settings
    fallback_to_faiss: bool = True  # Use FAISS scores if reranking fails
    min_score_threshold: float = 0.0  # Minimum similarity score to include
    
    def __post_init__(self):
        """Initialize similarity engine config if not provided."""
        if self.similarity_engine_config is None:
            self.similarity_engine_config = SimilarityEngineConfig(
                similarity_method=SimilarityMethod.HYBRID_WEIGHTED,
                enable_caching=self.enable_caching
            )


@dataclass
class RetrievalResult:
    """Result from two-stage retrieval."""
    doc_id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any]
    stage: str  # "faiss" or "quantum"
    debug_info: Optional[Dict[str, Any]] = None


class TwoStageRetriever:
    """
    Two-stage retrieval pipeline with FAISS and quantum reranking.
    
    Implements the complete retrieval flow:
    1. Fast initial retrieval using FAISS vector search
    2. Quantum-inspired reranking of top candidates
    3. Metadata and score preservation throughout
    """
    
    def __init__(self,
                 config: Optional[RetrieverConfig] = None,
                 embedding_processor: Optional[EmbeddingProcessor] = None):
        """
        Initialize two-stage retriever.
        
        Args:
            config: Retriever configuration
            embedding_processor: Processor for generating embeddings
        """
        self.config = config or RetrieverConfig()
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        
        # Initialize components
        self.faiss_store = QuantumFAISSStore(self.config.faiss_config)
        self.document_store = DocumentStore(self.embedding_processor)
        self.quantum_reranker = QuantumRAGReranker(self.config.similarity_engine_config)
        
        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "avg_faiss_time_ms": 0.0,
            "avg_rerank_time_ms": 0.0,
            "avg_total_time_ms": 0.0,
            "fallback_count": 0
        }
        
        logger.info(f"Two-stage retriever initialized: "
                   f"initial_k={self.config.initial_k}, "
                   f"final_k={self.config.final_k}, "
                   f"method={self.config.reranking_method}")
    
    def add_documents(self,
                     documents: List[Document],
                     generate_embeddings: bool = True) -> int:
        """
        Add documents to both stores.
        
        Args:
            documents: List of documents to add
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            Number of documents added
        """
        # Add to document store (generates embeddings if needed)
        added = self.document_store.add_documents(
            documents, 
            generate_embeddings=generate_embeddings,
            batch_size=self.config.batch_size
        )
        
        if added > 0:
            # Extract documents with embeddings
            docs_with_embeddings = []
            for doc in documents:
                if doc.embedding is not None:
                    docs_with_embeddings.append(doc)
            
            if docs_with_embeddings:
                # Prepare data for FAISS
                embeddings = np.array([doc.embedding for doc in docs_with_embeddings])
                doc_ids = [doc.doc_id for doc in docs_with_embeddings]
                metadatas = [doc.metadata.to_dict() for doc in docs_with_embeddings]
                
                # Add to FAISS
                self.faiss_store.add_documents(embeddings, doc_ids, metadatas)
                logger.info(f"Added {len(docs_with_embeddings)} documents to FAISS index")
        
        return added
    
    def add_texts(self,
                 texts: List[str],
                 metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add texts as documents.
        
        Args:
            texts: List of text contents
            metadatas: Optional metadata for each text
            
        Returns:
            List of document IDs
        """
        doc_ids = self.document_store.add_texts(texts, metadatas)
        
        # Get documents with embeddings and add to FAISS
        documents = [self.document_store.get_document(doc_id) for doc_id in doc_ids]
        self.add_documents(documents, generate_embeddings=False)
        
        return doc_ids
    
    def retrieve(self,
                query: str,
                k: Optional[int] = None,
                filter_dict: Optional[Dict[str, Any]] = None,
                return_debug_info: bool = False) -> List[RetrievalResult]:
        """
        Perform two-stage retrieval.
        
        Args:
            query: Query text
            k: Number of final results (overrides config.final_k)
            filter_dict: Metadata filters
            return_debug_info: Whether to include debug information
            
        Returns:
            List of retrieval results
        """
        start_time = time.time()
        k = k or self.config.final_k
        
        # Stage 1: FAISS retrieval
        faiss_results, faiss_time = self._faiss_retrieval(query, filter_dict)
        
        if not faiss_results:
            logger.warning("No results from FAISS retrieval")
            return []
        
        # Stage 2: Quantum reranking
        try:
            reranked_results, rerank_time = self._quantum_reranking(
                query, faiss_results, k
            )
            stage = "quantum"
        except Exception as e:
            logger.error(f"Quantum reranking failed: {e}")
            if self.config.fallback_to_faiss:
                logger.info("Falling back to FAISS scores")
                reranked_results = self._fallback_to_faiss(faiss_results, k)
                rerank_time = 0
                stage = "faiss"
                self.stats["fallback_count"] += 1
            else:
                raise
        
        # Build final results
        final_results = []
        for i, result in enumerate(reranked_results[:k]):
            # Get full document
            doc = self.document_store.get_document(result["doc_id"])
            if doc is None:
                logger.warning(f"Document {result['doc_id']} not found in store")
                continue
            
            # Build retrieval result
            retrieval_result = RetrievalResult(
                doc_id=doc.doc_id,
                content=doc.content,
                score=result["score"],
                rank=i + 1,
                metadata=doc.metadata.to_dict(),
                stage=stage
            )
            
            # Add debug info if requested
            if return_debug_info:
                retrieval_result.debug_info = {
                    "faiss_score": result.get("faiss_score"),
                    "faiss_rank": result.get("faiss_rank"),
                    "rerank_method": self.config.reranking_method,
                    "faiss_time_ms": faiss_time * 1000,
                    "rerank_time_ms": rerank_time * 1000
                }
            
            final_results.append(retrieval_result)
        
        # Update statistics
        total_time = time.time() - start_time
        self._update_stats(faiss_time, rerank_time, total_time)
        
        logger.info(f"Retrieved {len(final_results)} results in {total_time*1000:.2f}ms "
                   f"(FAISS: {faiss_time*1000:.2f}ms, Rerank: {rerank_time*1000:.2f}ms)")
        
        return final_results
    
    def _faiss_retrieval(self, 
                        query: str,
                        filter_dict: Optional[Dict[str, Any]]) -> Tuple[List[Dict], float]:
        """
        Perform initial FAISS retrieval.
        
        Returns:
            Tuple of (results, time_taken)
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_processor.encode_texts([query])[0]
        
        # Search FAISS
        search_results = self.faiss_store.search(
            query_embedding, 
            k=self.config.initial_k,
            filter_dict=filter_dict
        )
        
        # Convert to internal format
        results = []
        for i, sr in enumerate(search_results):
            results.append({
                "doc_id": sr.doc_id,
                "faiss_score": sr.score,
                "faiss_rank": i + 1,
                "metadata": sr.metadata
            })
        
        elapsed = time.time() - start_time
        return results, elapsed
    
    def _quantum_reranking(self,
                          query: str,
                          candidates: List[Dict],
                          k: int) -> Tuple[List[Dict], float]:
        """
        Perform quantum reranking on candidates.
        
        Returns:
            Tuple of (reranked_results, time_taken)
        """
        start_time = time.time()
        
        # Get candidate texts
        candidate_texts = []
        for candidate in candidates:
            doc = self.document_store.get_document(candidate["doc_id"])
            if doc:
                candidate_texts.append(doc.content)
            else:
                # Shouldn't happen, but handle gracefully
                candidate_texts.append("")
        
        # Perform quantum reranking
        reranked = self.quantum_reranker.rerank(
            query,
            candidate_texts,
            top_k=k,
            method=self.config.reranking_method
        )
        
        # Merge with original candidate info
        results = []
        for reranked_item in reranked:
            # Find original candidate by matching text
            for i, candidate in enumerate(candidates):
                doc = self.document_store.get_document(candidate["doc_id"])
                if doc and doc.content == reranked_item["text"]:
                    results.append({
                        "doc_id": candidate["doc_id"],
                        "score": reranked_item["similarity_score"],
                        "quantum_rank": reranked_item["rank"],
                        "faiss_score": candidate["faiss_score"],
                        "faiss_rank": candidate["faiss_rank"],
                        "metadata": candidate["metadata"]
                    })
                    break
        
        # Apply score threshold
        if self.config.min_score_threshold > 0:
            results = [r for r in results if r["score"] >= self.config.min_score_threshold]
        
        elapsed = time.time() - start_time
        return results, elapsed
    
    def _fallback_to_faiss(self, faiss_results: List[Dict], k: int) -> List[Dict]:
        """Fallback to FAISS scores if reranking fails."""
        # Convert FAISS scores to final format
        results = []
        for result in faiss_results[:k]:
            results.append({
                "doc_id": result["doc_id"],
                "score": result["faiss_score"],
                "faiss_score": result["faiss_score"],
                "faiss_rank": result["faiss_rank"],
                "metadata": result["metadata"]
            })
        return results
    
    def _update_stats(self, faiss_time: float, rerank_time: float, total_time: float):
        """Update performance statistics."""
        self.stats["total_queries"] += 1
        n = self.stats["total_queries"]
        
        # Update rolling averages
        self.stats["avg_faiss_time_ms"] = (
            (self.stats["avg_faiss_time_ms"] * (n - 1) + faiss_time * 1000) / n
        )
        self.stats["avg_rerank_time_ms"] = (
            (self.stats["avg_rerank_time_ms"] * (n - 1) + rerank_time * 1000) / n
        )
        self.stats["avg_total_time_ms"] = (
            (self.stats["avg_total_time_ms"] * (n - 1) + total_time * 1000) / n
        )
    
    def batch_retrieve(self,
                      queries: List[str],
                      k: Optional[int] = None) -> List[List[RetrievalResult]]:
        """
        Retrieve for multiple queries.
        
        Args:
            queries: List of query texts
            k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        results = []
        for query in queries:
            query_results = self.retrieve(query, k)
            results.append(query_results)
        return results
    
    def save_indices(self, path: str):
        """Save both FAISS index and document store."""
        from pathlib import Path
        
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = base_path / "faiss"
        self.faiss_store.save_index(str(faiss_path))
        
        # Save document store
        doc_store_path = base_path / "documents.pkl"
        self.document_store.save(str(doc_store_path))
        
        logger.info(f"Saved indices to {path}")
    
    def load_indices(self, path: str):
        """Load both FAISS index and document store."""
        from pathlib import Path
        
        base_path = Path(path)
        
        # Load FAISS index
        faiss_path = base_path / "faiss"
        self.faiss_store.load_index(str(faiss_path))
        
        # Load document store
        doc_store_path = base_path / "documents.pkl"
        self.document_store.load(str(doc_store_path))
        
        logger.info(f"Loaded indices from {path}")
    
    def clear(self):
        """Clear all data from both stores."""
        self.faiss_store.clear()
        self.document_store.clear()
        logger.info("Two-stage retriever cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "retriever_stats": self.stats,
            "faiss_stats": self.faiss_store.get_stats(),
            "document_stats": self.document_store.get_stats(),
            "quantum_stats": self.quantum_reranker.get_performance_stats(),
            "config": {
                "initial_k": self.config.initial_k,
                "final_k": self.config.final_k,
                "reranking_method": self.config.reranking_method,
                "fallback_enabled": self.config.fallback_to_faiss
            }
        }