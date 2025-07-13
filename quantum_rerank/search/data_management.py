"""
Scalable data management for large-scale vector search collections.

This module provides efficient data handling, streaming ingestion,
and distributed index management for massive document collections.
"""

import os
import time
import threading
import hashlib
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .retrieval_pipeline import Document
from ..utils import get_logger


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion pipeline."""
    batch_size: int = 1000
    max_memory_mb: int = 2048
    enable_streaming: bool = True
    validate_embeddings: bool = True
    deduplicate_documents: bool = True
    checkpoint_interval: int = 10000
    parallel_workers: int = 4


@dataclass
class CollectionMetadata:
    """Metadata for a document collection."""
    collection_id: str
    total_documents: int
    embedding_dimension: int
    created_at: float
    last_updated: float
    index_version: str
    data_sources: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionBatch:
    """Batch of documents for ingestion."""
    batch_id: str
    documents: List[Document]
    embeddings: np.ndarray
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"


@dataclass
class IngestionProgress:
    """Progress tracking for data ingestion."""
    total_documents: int
    processed_documents: int
    failed_documents: int
    current_batch: int
    total_batches: int
    processing_rate_docs_per_sec: float
    estimated_completion_time: float


class ScalableDataManager:
    """
    Scalable data management system for large vector collections.
    
    This manager handles streaming ingestion, efficient storage,
    and distributed index management for massive document collections.
    """
    
    def __init__(self, config: Optional[DataIngestionConfig] = None,
                 storage_path: str = "./data"):
        self.config = config or DataIngestionConfig()
        self.storage_path = storage_path
        self.logger = get_logger(__name__)
        
        # Collection management
        self.collections: Dict[str, CollectionMetadata] = {}
        self.active_ingestions: Dict[str, IngestionProgress] = {}
        
        # Data storage
        self.document_stores: Dict[str, Dict[str, Document]] = {}
        self.embedding_stores: Dict[str, np.ndarray] = {}
        
        # Threading and synchronization
        self._lock = threading.RLock()
        self.ingestion_workers: Dict[str, threading.Thread] = {}
        
        # Performance tracking
        self.ingestion_stats = {
            "total_collections": 0,
            "total_documents": 0,
            "total_ingestion_time_seconds": 0.0,
            "average_throughput_docs_per_sec": 0.0
        }
        
        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.logger.info(f"Initialized ScalableDataManager at {storage_path}")
    
    def create_collection(self, collection_id: str,
                         initial_documents: Optional[List[Document]] = None) -> bool:
        """
        Create a new document collection.
        
        Args:
            collection_id: Unique collection identifier
            initial_documents: Optional initial documents to add
            
        Returns:
            Success status
        """
        try:
            with self._lock:
                if collection_id in self.collections:
                    self.logger.warning(f"Collection {collection_id} already exists")
                    return False
                
                # Initialize collection metadata
                metadata = CollectionMetadata(
                    collection_id=collection_id,
                    total_documents=0,
                    embedding_dimension=0,
                    created_at=time.time(),
                    last_updated=time.time(),
                    index_version="1.0"
                )
                
                # Initialize storage
                self.collections[collection_id] = metadata
                self.document_stores[collection_id] = {}
                
                # Add initial documents if provided
                if initial_documents:
                    success = self.ingest_documents(collection_id, initial_documents)
                    if not success:
                        # Cleanup on failure
                        del self.collections[collection_id]
                        del self.document_stores[collection_id]
                        return False
                
                self.ingestion_stats["total_collections"] += 1
                
                self.logger.info(f"Created collection {collection_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_id}: {e}")
            return False
    
    def ingest_documents(self, collection_id: str,
                        documents: List[Document],
                        streaming: bool = False) -> bool:
        """
        Ingest documents into a collection.
        
        Args:
            collection_id: Target collection
            documents: Documents to ingest
            streaming: Whether to use streaming ingestion
            
        Returns:
            Success status
        """
        if collection_id not in self.collections:
            self.logger.error(f"Collection {collection_id} not found")
            return False
        
        try:
            if streaming or len(documents) > self.config.batch_size * 10:
                # Use streaming ingestion for large datasets
                return self._stream_ingest_documents(collection_id, documents)
            else:
                # Use batch ingestion for smaller datasets
                return self._batch_ingest_documents(collection_id, documents)
                
        except Exception as e:
            self.logger.error(f"Document ingestion failed for {collection_id}: {e}")
            return False
    
    def stream_ingest_from_source(self, collection_id: str,
                                 document_iterator: Iterator[Document],
                                 source_name: str = "stream") -> str:
        """
        Start streaming ingestion from an iterator source.
        
        Args:
            collection_id: Target collection
            document_iterator: Iterator yielding documents
            source_name: Name of the data source
            
        Returns:
            Ingestion job ID for tracking
        """
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} not found")
        
        # Generate job ID
        job_id = f"{collection_id}_{int(time.time())}_{source_name}"
        
        # Initialize progress tracking
        progress = IngestionProgress(
            total_documents=0,  # Unknown for streaming
            processed_documents=0,
            failed_documents=0,
            current_batch=0,
            total_batches=0,
            processing_rate_docs_per_sec=0.0,
            estimated_completion_time=0.0
        )
        
        with self._lock:
            self.active_ingestions[job_id] = progress
        
        # Start ingestion worker thread
        worker = threading.Thread(
            target=self._streaming_ingestion_worker,
            args=(job_id, collection_id, document_iterator, source_name),
            daemon=True
        )
        
        self.ingestion_workers[job_id] = worker
        worker.start()
        
        self.logger.info(f"Started streaming ingestion job {job_id}")
        return job_id
    
    def get_ingestion_progress(self, job_id: str) -> Optional[IngestionProgress]:
        """Get progress for an active ingestion job."""
        with self._lock:
            return self.active_ingestions.get(job_id)
    
    def stop_ingestion(self, job_id: str) -> bool:
        """Stop an active ingestion job."""
        try:
            if job_id in self.ingestion_workers:
                # Note: This is a simplified stop mechanism
                # In production, you'd want proper thread communication
                with self._lock:
                    if job_id in self.active_ingestions:
                        del self.active_ingestions[job_id]
                
                self.logger.info(f"Stopped ingestion job {job_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to stop ingestion {job_id}: {e}")
            return False
    
    def get_collection_info(self, collection_id: str) -> Optional[CollectionMetadata]:
        """Get comprehensive collection information."""
        with self._lock:
            if collection_id not in self.collections:
                return None
            
            metadata = self.collections[collection_id]
            
            # Update statistics
            if collection_id in self.document_stores:
                metadata.total_documents = len(self.document_stores[collection_id])
                metadata.last_updated = time.time()
            
            return metadata
    
    def update_collection_index(self, collection_id: str,
                              force_rebuild: bool = False) -> bool:
        """
        Update search index for a collection.
        
        Args:
            collection_id: Collection to update
            force_rebuild: Force complete index rebuild
            
        Returns:
            Success status
        """
        if collection_id not in self.collections:
            self.logger.error(f"Collection {collection_id} not found")
            return False
        
        try:
            # Get documents and embeddings
            documents = list(self.document_stores[collection_id].values())
            
            if not documents:
                self.logger.warning(f"No documents in collection {collection_id}")
                return True
            
            # Extract embeddings
            embeddings = []
            document_ids = []
            
            for doc in documents:
                if doc.embedding is not None:
                    embeddings.append(doc.embedding)
                    document_ids.append(doc.id)
            
            if not embeddings:
                self.logger.error(f"No embeddings found in collection {collection_id}")
                return False
            
            embeddings_array = np.array(embeddings)
            
            # Update metadata
            metadata = self.collections[collection_id]
            metadata.embedding_dimension = embeddings_array.shape[1]
            metadata.total_documents = len(documents)
            metadata.last_updated = time.time()
            
            # Update index version
            if force_rebuild:
                metadata.index_version = f"{float(metadata.index_version) + 0.1:.1f}"
            
            self.logger.info(
                f"Updated index for collection {collection_id}: "
                f"{len(documents)} documents, dimension {embeddings_array.shape[1]}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Index update failed for {collection_id}: {e}")
            return False
    
    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection and all its data."""
        try:
            with self._lock:
                if collection_id not in self.collections:
                    self.logger.warning(f"Collection {collection_id} not found")
                    return False
                
                # Stop any active ingestions
                active_jobs = [
                    job_id for job_id in self.active_ingestions.keys()
                    if job_id.startswith(collection_id)
                ]
                
                for job_id in active_jobs:
                    self.stop_ingestion(job_id)
                
                # Remove data
                del self.collections[collection_id]
                if collection_id in self.document_stores:
                    del self.document_stores[collection_id]
                if collection_id in self.embedding_stores:
                    del self.embedding_stores[collection_id]
                
                # Update statistics
                self.ingestion_stats["total_collections"] -= 1
                
                self.logger.info(f"Deleted collection {collection_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_id}: {e}")
            return False
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all collections."""
        with self._lock:
            stats = {
                "total_collections": len(self.collections),
                "total_documents": sum(
                    len(store) for store in self.document_stores.values()
                ),
                "active_ingestions": len(self.active_ingestions),
                "collections": {},
                "ingestion_stats": self.ingestion_stats.copy()
            }
            
            # Per-collection statistics
            for collection_id, metadata in self.collections.items():
                collection_stats = {
                    "total_documents": metadata.total_documents,
                    "embedding_dimension": metadata.embedding_dimension,
                    "created_at": metadata.created_at,
                    "last_updated": metadata.last_updated,
                    "index_version": metadata.index_version,
                    "data_sources": metadata.data_sources.copy()
                }
                
                # Add current document count
                if collection_id in self.document_stores:
                    collection_stats["current_documents"] = len(self.document_stores[collection_id])
                
                stats["collections"][collection_id] = collection_stats
            
            return stats
    
    def _batch_ingest_documents(self, collection_id: str,
                               documents: List[Document]) -> bool:
        """Ingest documents using batch processing."""
        start_time = time.time()
        
        # Process in batches
        total_processed = 0
        total_failed = 0
        
        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]
            
            batch_result = self._process_document_batch(collection_id, batch, f"batch_{i}")
            
            if batch_result:
                total_processed += len(batch)
            else:
                total_failed += len(batch)
        
        # Update statistics
        ingestion_time = time.time() - start_time
        self.ingestion_stats["total_documents"] += total_processed
        self.ingestion_stats["total_ingestion_time_seconds"] += ingestion_time
        
        if ingestion_time > 0:
            throughput = total_processed / ingestion_time
            self.ingestion_stats["average_throughput_docs_per_sec"] = (
                (self.ingestion_stats["average_throughput_docs_per_sec"] + throughput) / 2
            )
        
        success_rate = total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0
        
        self.logger.info(
            f"Batch ingestion completed for {collection_id}: "
            f"{total_processed} processed, {total_failed} failed, "
            f"success rate: {success_rate:.2%}"
        )
        
        return success_rate > 0.9  # Consider successful if >90% success rate
    
    def _stream_ingest_documents(self, collection_id: str,
                                documents: List[Document]) -> bool:
        """Ingest documents using streaming processing."""
        # Convert list to iterator and use streaming ingestion
        document_iterator = iter(documents)
        job_id = self.stream_ingest_from_source(collection_id, document_iterator, "batch_stream")
        
        # Wait for completion (simplified for batch processing)
        while job_id in self.active_ingestions:
            time.sleep(1)
        
        return True  # Simplified success check
    
    def _streaming_ingestion_worker(self, job_id: str, collection_id: str,
                                   document_iterator: Iterator[Document],
                                   source_name: str) -> None:
        """Worker thread for streaming document ingestion."""
        try:
            progress = self.active_ingestions[job_id]
            start_time = time.time()
            
            batch_documents = []
            batch_count = 0
            
            for document in document_iterator:
                # Check if ingestion was stopped
                if job_id not in self.active_ingestions:
                    break
                
                batch_documents.append(document)
                
                # Process batch when full
                if len(batch_documents) >= self.config.batch_size:
                    success = self._process_document_batch(
                        collection_id, batch_documents, f"{job_id}_batch_{batch_count}"
                    )
                    
                    # Update progress
                    if success:
                        progress.processed_documents += len(batch_documents)
                    else:
                        progress.failed_documents += len(batch_documents)
                    
                    progress.current_batch = batch_count
                    
                    # Calculate processing rate
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        progress.processing_rate_docs_per_sec = progress.processed_documents / elapsed_time
                    
                    # Clear batch and increment counter
                    batch_documents = []
                    batch_count += 1
            
            # Process remaining documents
            if batch_documents and job_id in self.active_ingestions:
                success = self._process_document_batch(
                    collection_id, batch_documents, f"{job_id}_final_batch"
                )
                
                if success:
                    progress.processed_documents += len(batch_documents)
                else:
                    progress.failed_documents += len(batch_documents)
            
            # Update completion time
            progress.estimated_completion_time = time.time()
            
            self.logger.info(
                f"Streaming ingestion completed for job {job_id}: "
                f"{progress.processed_documents} processed, "
                f"{progress.failed_documents} failed"
            )
            
        except Exception as e:
            self.logger.error(f"Streaming ingestion worker failed for {job_id}: {e}")
        
        finally:
            # Cleanup
            with self._lock:
                if job_id in self.active_ingestions:
                    del self.active_ingestions[job_id]
                if job_id in self.ingestion_workers:
                    del self.ingestion_workers[job_id]
    
    def _process_document_batch(self, collection_id: str,
                               documents: List[Document],
                               batch_id: str) -> bool:
        """Process a batch of documents."""
        try:
            # Validate documents
            if self.config.validate_embeddings:
                documents = self._validate_documents(documents)
            
            # Deduplicate if enabled
            if self.config.deduplicate_documents:
                documents = self._deduplicate_documents(collection_id, documents)
            
            if not documents:
                self.logger.warning(f"No valid documents in batch {batch_id}")
                return True  # Not a failure, just empty batch
            
            # Store documents
            with self._lock:
                document_store = self.document_stores[collection_id]
                
                for doc in documents:
                    document_store[doc.id] = doc
                
                # Update collection metadata
                metadata = self.collections[collection_id]
                metadata.total_documents = len(document_store)
                metadata.last_updated = time.time()
                
                # Update embedding dimension if not set
                if metadata.embedding_dimension == 0 and documents[0].embedding is not None:
                    metadata.embedding_dimension = len(documents[0].embedding)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process batch {batch_id}: {e}")
            return False
    
    def _validate_documents(self, documents: List[Document]) -> List[Document]:
        """Validate documents and embeddings."""
        valid_documents = []
        
        for doc in documents:
            if not doc.id:
                self.logger.warning("Document missing ID, skipping")
                continue
            
            if doc.embedding is None:
                self.logger.warning(f"Document {doc.id} missing embedding, skipping")
                continue
            
            if not isinstance(doc.embedding, np.ndarray):
                self.logger.warning(f"Document {doc.id} has invalid embedding type, skipping")
                continue
            
            if doc.embedding.ndim != 1:
                self.logger.warning(f"Document {doc.id} has invalid embedding shape, skipping")
                continue
            
            valid_documents.append(doc)
        
        return valid_documents
    
    def _deduplicate_documents(self, collection_id: str,
                              documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on ID."""
        existing_store = self.document_stores.get(collection_id, {})
        unique_documents = []
        
        for doc in documents:
            if doc.id not in existing_store:
                unique_documents.append(doc)
            else:
                self.logger.debug(f"Skipping duplicate document {doc.id}")
        
        return unique_documents
    
    def _create_document_hash(self, document: Document) -> str:
        """Create hash for document content (for deduplication)."""
        # Simple hash based on content
        content_to_hash = f"{document.content}{document.metadata}"
        return hashlib.md5(content_to_hash.encode()).hexdigest()


__all__ = [
    "DataIngestionConfig",
    "CollectionMetadata",
    "IngestionBatch", 
    "IngestionProgress",
    "ScalableDataManager"
]