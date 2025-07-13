"""
FAISS vector store for efficient similarity search.

This module implements a FAISS-based vector database for initial candidate
retrieval in the two-stage quantum reranking pipeline.

Implements PRD Section 2.2: Vector Search (FAISS) and Section 5.2: RAG Pipeline Integration.
"""

import faiss
import numpy as np
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
from enum import Enum

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Available FAISS index types."""
    FLAT = "Flat"  # Exact search, no training required
    IVF = "IVF"    # Inverted file index, requires training
    HNSW = "HNSW"  # Hierarchical Navigable Small World graph
    LSH = "LSH"    # Locality Sensitive Hashing


@dataclass
class FAISSConfig:
    """Configuration for FAISS vector store."""
    index_type: IndexType = IndexType.FLAT
    dimension: int = 768  # Default for sentence-transformers
    
    # IVF specific parameters
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    
    # HNSW specific parameters
    M: int = 32  # Number of connections per node
    ef_construction: int = 200  # Size of dynamic candidate list
    ef_search: int = 50  # Size of dynamic candidate list for search
    
    # LSH specific parameters
    nbits: int = 768  # Number of bits for LSH
    
    # General parameters
    use_gpu: bool = False
    normalize_vectors: bool = True
    metric: str = "cosine"  # cosine or l2
    

@dataclass
class SearchResult:
    """Result from a FAISS search operation."""
    doc_id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[np.ndarray] = None


class QuantumFAISSStore:
    """
    FAISS-based vector store for efficient similarity search.
    
    Supports multiple index types and provides persistence capabilities
    for integration with the quantum reranking pipeline.
    """
    
    def __init__(self, config: FAISSConfig = None):
        """
        Initialize FAISS vector store.
        
        Args:
            config: Configuration for FAISS index
        """
        self.config = config or FAISSConfig()
        self.index = None
        self.id_to_metadata: Dict[int, Dict] = {}
        self.doc_id_to_index: Dict[str, int] = {}
        self.next_index = 0
        
        # Performance tracking
        self.stats = {
            'total_documents': 0,
            'total_searches': 0,
            'avg_search_time_ms': 0.0,
            'index_built': False
        }
        
        self._initialize_index()
        logger.info(f"FAISS store initialized with {self.config.index_type.value} index")
    
    def _initialize_index(self):
        """Initialize the FAISS index based on configuration."""
        d = self.config.dimension
        
        if self.config.index_type == IndexType.FLAT:
            if self.config.metric == "cosine":
                self.index = faiss.IndexFlatIP(d)  # Inner product for cosine
            else:
                self.index = faiss.IndexFlatL2(d)  # L2 distance
                
        elif self.config.index_type == IndexType.IVF:
            quantizer = faiss.IndexFlatL2(d)
            if self.config.metric == "cosine":
                self.index = faiss.IndexIVFFlat(quantizer, d, self.config.nlist, 
                                               faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexIVFFlat(quantizer, d, self.config.nlist)
                
        elif self.config.index_type == IndexType.HNSW:
            self.index = faiss.IndexHNSWFlat(d, self.config.M)
            self.index.hnsw.efConstruction = self.config.ef_construction
            self.index.hnsw.efSearch = self.config.ef_search
            
        elif self.config.index_type == IndexType.LSH:
            self.index = faiss.IndexLSH(d, self.config.nbits)
            
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        # Move to GPU if requested and available
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            logger.info("FAISS index moved to GPU")
    
    def add_documents(self, 
                     embeddings: np.ndarray,
                     doc_ids: List[str],
                     metadata: Optional[List[Dict]] = None) -> int:
        """
        Add documents to the FAISS index.
        
        Args:
            embeddings: Document embeddings as numpy array (n_docs, dimension)
            doc_ids: Unique document identifiers
            metadata: Optional metadata for each document
            
        Returns:
            Number of documents added
        """
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError("Number of embeddings must match number of doc_ids")
        
        if embeddings.shape[1] != self.config.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match "
                           f"configured dimension {self.config.dimension}")
        
        # Normalize vectors if configured
        if self.config.normalize_vectors:
            embeddings = self._normalize_vectors(embeddings)
        
        # Train index if needed (for IVF)
        if (self.config.index_type == IndexType.IVF and 
            not self.stats['index_built'] and 
            embeddings.shape[0] >= self.config.nlist):
            logger.info(f"Training IVF index with {embeddings.shape[0]} vectors")
            self.index.train(embeddings)
            self.stats['index_built'] = True
        
        # Add to index
        start_idx = self.next_index
        self.index.add(embeddings)
        
        # Store metadata and mappings
        if metadata is None:
            metadata = [{}] * len(doc_ids)
            
        for i, (doc_id, meta) in enumerate(zip(doc_ids, metadata)):
            idx = start_idx + i
            self.doc_id_to_index[doc_id] = idx
            self.id_to_metadata[idx] = {
                'doc_id': doc_id,
                'metadata': meta
            }
        
        self.next_index += len(doc_ids)
        self.stats['total_documents'] += len(doc_ids)
        
        logger.info(f"Added {len(doc_ids)} documents to FAISS index "
                   f"(total: {self.stats['total_documents']})")
        
        return len(doc_ids)
    
    def search(self, 
               query_embedding: np.ndarray,
               k: int = 100,
               filter_dict: Optional[Dict] = None) -> List[SearchResult]:
        """
        Search for similar documents in the FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_dict: Optional metadata filters (post-processing)
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        start_time = time.time()
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize if configured
        if self.config.normalize_vectors:
            query_embedding = self._normalize_vectors(query_embedding)
        
        # Adjust k if we have fewer documents
        k = min(k, self.stats['total_documents'])
        if k == 0:
            return []
        
        # Search
        if self.config.index_type == IndexType.IVF:
            self.index.nprobe = self.config.nprobe
            
        distances, indices = self.index.search(query_embedding, k)
        
        # Process results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
                
            if idx in self.id_to_metadata:
                meta_info = self.id_to_metadata[idx]
                
                # Apply filters if provided
                if filter_dict:
                    match = all(
                        meta_info['metadata'].get(key) == value 
                        for key, value in filter_dict.items()
                    )
                    if not match:
                        continue
                
                # Convert distance to similarity score
                if self.config.metric == "cosine":
                    score = float(dist)  # Inner product is similarity for normalized vectors
                else:
                    score = 1.0 / (1.0 + float(dist))  # Convert L2 distance to similarity
                
                results.append(SearchResult(
                    doc_id=meta_info['doc_id'],
                    score=score,
                    metadata=meta_info['metadata']
                ))
        
        # Update stats
        search_time = time.time() - start_time
        self._update_search_stats(search_time)
        
        logger.debug(f"FAISS search returned {len(results)} results in {search_time*1000:.2f}ms")
        
        return results
    
    def batch_search(self,
                    query_embeddings: np.ndarray,
                    k: int = 100) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Multiple query embeddings (n_queries, dimension)
            k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Normalize if configured
        if self.config.normalize_vectors:
            query_embeddings = self._normalize_vectors(query_embeddings)
        
        # Adjust k
        k = min(k, self.stats['total_documents'])
        if k == 0:
            return [[] for _ in range(query_embeddings.shape[0])]
        
        # Batch search
        start_time = time.time()
        distances, indices = self.index.search(query_embeddings, k)
        
        # Process results for each query
        all_results = []
        for query_idx in range(query_embeddings.shape[0]):
            results = []
            for dist, idx in zip(distances[query_idx], indices[query_idx]):
                if idx < 0 or idx not in self.id_to_metadata:
                    continue
                
                meta_info = self.id_to_metadata[idx]
                
                # Convert distance to similarity
                if self.config.metric == "cosine":
                    score = float(dist)
                else:
                    score = 1.0 / (1.0 + float(dist))
                
                results.append(SearchResult(
                    doc_id=meta_info['doc_id'],
                    score=score,
                    metadata=meta_info['metadata']
                ))
            
            all_results.append(results)
        
        # Update stats
        search_time = time.time() - start_time
        self._update_search_stats(search_time, batch_size=len(query_embeddings))
        
        return all_results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document metadata by ID."""
        if doc_id not in self.doc_id_to_index:
            return None
            
        idx = self.doc_id_to_index[doc_id]
        return self.id_to_metadata.get(idx)
    
    def remove_documents(self, doc_ids: List[str]) -> int:
        """
        Remove documents from the index.
        
        Note: FAISS doesn't support direct removal, so this marks documents
        as deleted in metadata. A rebuild may be needed for actual removal.
        """
        removed = 0
        for doc_id in doc_ids:
            if doc_id in self.doc_id_to_index:
                idx = self.doc_id_to_index[doc_id]
                del self.doc_id_to_index[doc_id]
                del self.id_to_metadata[idx]
                removed += 1
        
        self.stats['total_documents'] -= removed
        logger.info(f"Marked {removed} documents for removal")
        
        return removed
    
    def save_index(self, path: str):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            path: Directory path to save index files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = path / "metadata.pkl"
        metadata = {
            'id_to_metadata': self.id_to_metadata,
            'doc_id_to_index': self.doc_id_to_index,
            'next_index': self.next_index,
            'stats': self.stats,
            'config': self.config
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved FAISS index to {path}")
    
    def load_index(self, path: str):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            path: Directory path containing index files
        """
        path = Path(path)
        
        # Load FAISS index
        index_path = path / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.id_to_metadata = metadata['id_to_metadata']
        self.doc_id_to_index = metadata['doc_id_to_index']
        self.next_index = metadata['next_index']
        self.stats = metadata['stats']
        self.config = metadata['config']
        
        # Move to GPU if needed
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        logger.info(f"Loaded FAISS index from {path} "
                   f"({self.stats['total_documents']} documents)")
    
    def clear(self):
        """Clear all documents from the index."""
        self._initialize_index()
        self.id_to_metadata.clear()
        self.doc_id_to_index.clear()
        self.next_index = 0
        self.stats['total_documents'] = 0
        self.stats['index_built'] = False
        logger.info("FAISS index cleared")
    
    def get_stats(self) -> Dict:
        """Get performance and usage statistics."""
        return {
            **self.stats,
            'index_type': self.config.index_type.value,
            'dimension': self.config.dimension,
            'use_gpu': self.config.use_gpu,
            'metric': self.config.metric
        }
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.config.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        return vectors
    
    def _update_search_stats(self, search_time: float, batch_size: int = 1):
        """Update search performance statistics."""
        self.stats['total_searches'] += batch_size
        
        # Update rolling average
        n = self.stats['total_searches']
        current_avg = self.stats['avg_search_time_ms']
        new_time_ms = (search_time * 1000) / batch_size  # Per-query time
        
        self.stats['avg_search_time_ms'] = (
            (current_avg * (n - batch_size) + new_time_ms * batch_size) / n
        )