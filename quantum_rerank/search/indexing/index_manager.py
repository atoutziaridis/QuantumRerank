"""
Dynamic index management for scalable vector search.

This module provides intelligent index management with automatic optimization,
version control, and efficient incremental updates.
"""

import os
import time
import threading
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..backends.base_backend import BaseVectorSearchBackend
from ..backends.faiss_backend import FAISSBackend
from ...utils import get_logger


class IndexStatus(Enum):
    """Index status states."""
    BUILDING = "building"
    READY = "ready"
    UPDATING = "updating"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    DEPRECATED = "deprecated"


@dataclass
class IndexConfiguration:
    """Configuration for vector index."""
    index_type: str
    dimension: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    backend_type: str = "faiss"
    enable_gpu: bool = False
    memory_limit_mb: int = 1024
    optimization_enabled: bool = True


@dataclass
class IndexMetadata:
    """Metadata for a vector index."""
    index_id: str
    collection_id: str
    version: str
    status: IndexStatus
    configuration: IndexConfiguration
    created_at: float
    last_updated: float
    last_optimized: float = 0.0
    document_count: int = 0
    build_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    performance_stats: Dict[str, float] = field(default_factory=dict)


@dataclass 
class IndexUpdateJob:
    """Index update job tracking."""
    job_id: str
    index_id: str
    operation_type: str  # "incremental", "rebuild", "optimize"
    documents_to_add: List[str] = field(default_factory=list)
    documents_to_remove: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    progress: float = 0.0
    status: str = "pending"
    error_message: Optional[str] = None


class DynamicIndexManager:
    """
    Dynamic index management system for scalable vector search.
    
    This manager provides automated index lifecycle management including
    creation, optimization, updates, and version control.
    """
    
    def __init__(self, storage_path: str = "./indexes"):
        self.storage_path = storage_path
        self.logger = get_logger(__name__)
        
        # Index management
        self.indexes: Dict[str, BaseVectorSearchBackend] = {}
        self.index_metadata: Dict[str, IndexMetadata] = {}
        self.active_jobs: Dict[str, IndexUpdateJob] = {}
        
        # Configuration
        self.default_configurations = self._initialize_default_configurations()
        self.optimization_thresholds = self._initialize_optimization_thresholds()
        
        # Threading
        self._lock = threading.RLock()
        self.worker_threads: Dict[str, threading.Thread] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.logger.info(f"Initialized DynamicIndexManager at {storage_path}")
    
    def create_index(self, collection_id: str,
                    embeddings: np.ndarray,
                    document_ids: List[str],
                    config: Optional[IndexConfiguration] = None) -> Optional[str]:
        """
        Create a new vector index for a collection.
        
        Args:
            collection_id: Collection identifier
            embeddings: Embedding vectors to index
            document_ids: Corresponding document identifiers
            config: Index configuration (uses auto-selected if None)
            
        Returns:
            Index ID if successful, None otherwise
        """
        try:
            # Generate index ID
            index_id = f"{collection_id}_{int(time.time())}"
            
            # Auto-select configuration if not provided
            if config is None:
                config = self._select_optimal_configuration(embeddings.shape)
            
            # Create index metadata
            metadata = IndexMetadata(
                index_id=index_id,
                collection_id=collection_id,
                version="1.0",
                status=IndexStatus.BUILDING,
                configuration=config,
                created_at=time.time(),
                last_updated=time.time(),
                document_count=len(document_ids)
            )
            
            with self._lock:
                self.index_metadata[index_id] = metadata
            
            # Build index asynchronously
            job_id = self._start_index_build_job(index_id, embeddings, document_ids)
            
            self.logger.info(f"Started building index {index_id} (job: {job_id})")
            return index_id
            
        except Exception as e:
            self.logger.error(f"Failed to create index for collection {collection_id}: {e}")
            return None
    
    def get_or_create_index(self, collection_id: str,
                           embeddings: Optional[np.ndarray] = None,
                           document_ids: Optional[List[str]] = None) -> Optional[str]:
        """
        Get existing index for collection or create new one.
        
        Args:
            collection_id: Collection identifier
            embeddings: Embedding vectors (required for creation)
            document_ids: Document identifiers (required for creation)
            
        Returns:
            Index ID if available/created, None otherwise
        """
        # Look for existing ready index
        existing_index = self._find_ready_index(collection_id)
        if existing_index:
            return existing_index
        
        # Create new index if embeddings provided
        if embeddings is not None and document_ids is not None:
            return self.create_index(collection_id, embeddings, document_ids)
        
        self.logger.warning(f"No ready index found for collection {collection_id} and no data provided for creation")
        return None
    
    def update_index(self, index_id: str,
                    new_embeddings: Optional[np.ndarray] = None,
                    new_document_ids: Optional[List[str]] = None,
                    remove_document_ids: Optional[List[str]] = None,
                    force_rebuild: bool = False) -> Optional[str]:
        """
        Update existing index with new data.
        
        Args:
            index_id: Index to update
            new_embeddings: New embeddings to add
            new_document_ids: New document IDs to add
            remove_document_ids: Document IDs to remove
            force_rebuild: Force complete rebuild instead of incremental update
            
        Returns:
            Update job ID if started, None otherwise
        """
        if index_id not in self.index_metadata:
            self.logger.error(f"Index {index_id} not found")
            return None
        
        try:
            # Determine update strategy
            if force_rebuild or self._should_rebuild_index(index_id, new_embeddings):
                operation_type = "rebuild"
            else:
                operation_type = "incremental"
            
            # Create update job
            job_id = f"{index_id}_update_{int(time.time())}"
            job = IndexUpdateJob(
                job_id=job_id,
                index_id=index_id,
                operation_type=operation_type,
                documents_to_add=new_document_ids or [],
                documents_to_remove=remove_document_ids or []
            )
            
            with self._lock:
                self.active_jobs[job_id] = job
                # Mark index as updating
                self.index_metadata[index_id].status = IndexStatus.UPDATING
            
            # Start update worker
            worker = threading.Thread(
                target=self._index_update_worker,
                args=(job_id, new_embeddings, new_document_ids, remove_document_ids),
                daemon=True
            )
            
            self.worker_threads[job_id] = worker
            worker.start()
            
            self.logger.info(f"Started {operation_type} update for index {index_id} (job: {job_id})")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to start update for index {index_id}: {e}")
            return None
    
    def optimize_index(self, index_id: str) -> Optional[str]:
        """
        Optimize index performance.
        
        Args:
            index_id: Index to optimize
            
        Returns:
            Optimization job ID if started, None otherwise
        """
        if index_id not in self.index_metadata:
            self.logger.error(f"Index {index_id} not found")
            return None
        
        if index_id not in self.indexes:
            self.logger.error(f"Index {index_id} not loaded")
            return None
        
        try:
            # Create optimization job
            job_id = f"{index_id}_optimize_{int(time.time())}"
            job = IndexUpdateJob(
                job_id=job_id,
                index_id=index_id,
                operation_type="optimize"
            )
            
            with self._lock:
                self.active_jobs[job_id] = job
                self.index_metadata[index_id].status = IndexStatus.OPTIMIZING
            
            # Start optimization worker
            worker = threading.Thread(
                target=self._index_optimization_worker,
                args=(job_id,),
                daemon=True
            )
            
            self.worker_threads[job_id] = worker
            worker.start()
            
            self.logger.info(f"Started optimization for index {index_id} (job: {job_id})")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to start optimization for index {index_id}: {e}")
            return None
    
    def get_index(self, index_id: str) -> Optional[BaseVectorSearchBackend]:
        """Get loaded index backend."""
        with self._lock:
            return self.indexes.get(index_id)
    
    def load_index(self, index_id: str) -> bool:
        """Load index into memory."""
        if index_id not in self.index_metadata:
            self.logger.error(f"Index {index_id} not found")
            return False
        
        if index_id in self.indexes:
            return True  # Already loaded
        
        try:
            metadata = self.index_metadata[index_id]
            config = metadata.configuration
            
            # Create backend
            if config.backend_type == "faiss":
                backend_config = config.parameters.copy()
                backend_config.update({
                    "index_type": config.index_type,
                    "use_gpu": config.enable_gpu
                })
                backend = FAISSBackend(backend_config)
            else:
                raise ValueError(f"Unsupported backend type: {config.backend_type}")
            
            # Load index data
            index_path = self._get_index_path(index_id)
            if os.path.exists(index_path):
                success = backend.load_index(index_path)
                if not success:
                    raise RuntimeError("Failed to load index data")
            else:
                self.logger.warning(f"Index file not found for {index_id}, index may need rebuilding")
            
            with self._lock:
                self.indexes[index_id] = backend
                metadata.status = IndexStatus.READY
            
            self.logger.info(f"Loaded index {index_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index {index_id}: {e}")
            with self._lock:
                if index_id in self.index_metadata:
                    self.index_metadata[index_id].status = IndexStatus.ERROR
            return False
    
    def unload_index(self, index_id: str) -> bool:
        """Unload index from memory."""
        try:
            with self._lock:
                if index_id in self.indexes:
                    del self.indexes[index_id]
                    self.logger.info(f"Unloaded index {index_id}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to unload index {index_id}: {e}")
            return False
    
    def save_index(self, index_id: str) -> bool:
        """Save index to persistent storage."""
        if index_id not in self.indexes:
            self.logger.error(f"Index {index_id} not loaded")
            return False
        
        try:
            backend = self.indexes[index_id]
            index_path = self._get_index_path(index_id)
            
            success = backend.save_index(index_path)
            if success:
                # Save metadata
                metadata_path = self._get_metadata_path(index_id)
                self._save_metadata(index_id, metadata_path)
                
                self.logger.info(f"Saved index {index_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to save index {index_id}: {e}")
            return False
    
    def delete_index(self, index_id: str) -> bool:
        """Delete index and all associated data."""
        try:
            # Stop any active jobs for this index
            jobs_to_stop = [
                job_id for job_id, job in self.active_jobs.items()
                if job.index_id == index_id
            ]
            
            for job_id in jobs_to_stop:
                self._stop_job(job_id)
            
            # Unload from memory
            self.unload_index(index_id)
            
            # Remove files
            index_path = self._get_index_path(index_id)
            metadata_path = self._get_metadata_path(index_id)
            
            for path in [index_path, metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            # Remove from tracking
            with self._lock:
                if index_id in self.index_metadata:
                    del self.index_metadata[index_id]
                if index_id in self.performance_history:
                    del self.performance_history[index_id]
            
            self.logger.info(f"Deleted index {index_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete index {index_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[IndexUpdateJob]:
        """Get status of an index operation job."""
        with self._lock:
            return self.active_jobs.get(job_id)
    
    def get_index_info(self, index_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive index information."""
        if index_id not in self.index_metadata:
            return None
        
        metadata = self.index_metadata[index_id]
        
        info = {
            "index_id": index_id,
            "collection_id": metadata.collection_id,
            "version": metadata.version,
            "status": metadata.status.value,
            "configuration": metadata.configuration.__dict__,
            "created_at": metadata.created_at,
            "last_updated": metadata.last_updated,
            "last_optimized": metadata.last_optimized,
            "document_count": metadata.document_count,
            "build_time_ms": metadata.build_time_ms,
            "memory_usage_mb": metadata.memory_usage_mb,
            "performance_stats": metadata.performance_stats.copy(),
            "is_loaded": index_id in self.indexes
        }
        
        # Add backend statistics if loaded
        if index_id in self.indexes:
            backend_stats = self.indexes[index_id].get_statistics()
            info["backend_stats"] = backend_stats
        
        return info
    
    def list_indexes(self, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indexes, optionally filtered by collection."""
        indexes = []
        
        for index_id, metadata in self.index_metadata.items():
            if collection_id is None or metadata.collection_id == collection_id:
                info = self.get_index_info(index_id)
                if info:
                    indexes.append(info)
        
        return indexes
    
    def _start_index_build_job(self, index_id: str,
                              embeddings: np.ndarray,
                              document_ids: List[str]) -> str:
        """Start asynchronous index building job."""
        job_id = f"{index_id}_build"
        job = IndexUpdateJob(
            job_id=job_id,
            index_id=index_id,
            operation_type="build",
            documents_to_add=document_ids
        )
        
        with self._lock:
            self.active_jobs[job_id] = job
        
        # Start build worker
        worker = threading.Thread(
            target=self._index_build_worker,
            args=(job_id, embeddings, document_ids),
            daemon=True
        )
        
        self.worker_threads[job_id] = worker
        worker.start()
        
        return job_id
    
    def _index_build_worker(self, job_id: str,
                           embeddings: np.ndarray,
                           document_ids: List[str]) -> None:
        """Worker thread for building indexes."""
        try:
            job = self.active_jobs[job_id]
            index_id = job.index_id
            metadata = self.index_metadata[index_id]
            
            job.status = "building"
            job.progress = 0.1
            
            # Create backend
            config = metadata.configuration
            if config.backend_type == "faiss":
                backend_config = config.parameters.copy()
                backend_config.update({
                    "index_type": config.index_type,
                    "use_gpu": config.enable_gpu
                })
                backend = FAISSBackend(backend_config)
            else:
                raise ValueError(f"Unsupported backend type: {config.backend_type}")
            
            job.progress = 0.3
            
            # Build index
            start_time = time.time()
            backend.build_index(embeddings, document_ids)
            build_time_ms = (time.time() - start_time) * 1000
            
            job.progress = 0.8
            
            # Store index
            with self._lock:
                self.indexes[index_id] = backend
                
                # Update metadata
                metadata.status = IndexStatus.READY
                metadata.last_updated = time.time()
                metadata.build_time_ms = build_time_ms
                metadata.memory_usage_mb = backend.get_memory_usage_mb()
            
            job.progress = 0.9
            
            # Save to disk
            self.save_index(index_id)
            
            job.progress = 1.0
            job.status = "completed"
            
            self.logger.info(f"Index build completed for {index_id} ({build_time_ms:.1f}ms)")
            
        except Exception as e:
            self.logger.error(f"Index build failed for job {job_id}: {e}")
            job.status = "failed"
            job.error_message = str(e)
            
            # Mark index as error
            if job_id in self.active_jobs:
                index_id = self.active_jobs[job_id].index_id
                if index_id in self.index_metadata:
                    self.index_metadata[index_id].status = IndexStatus.ERROR
        
        finally:
            # Cleanup
            self._cleanup_job(job_id)
    
    def _index_update_worker(self, job_id: str,
                            new_embeddings: Optional[np.ndarray],
                            new_document_ids: Optional[List[str]],
                            remove_document_ids: Optional[List[str]]) -> None:
        """Worker thread for updating indexes."""
        try:
            job = self.active_jobs[job_id]
            index_id = job.index_id
            
            if index_id not in self.indexes:
                if not self.load_index(index_id):
                    raise RuntimeError(f"Failed to load index {index_id}")
            
            backend = self.indexes[index_id]
            job.status = "updating"
            job.progress = 0.1
            
            # Handle removals
            if remove_document_ids:
                backend.remove_embeddings(remove_document_ids)
                job.progress = 0.4
            
            # Handle additions
            if new_embeddings is not None and new_document_ids is not None:
                backend.add_embeddings(new_embeddings, new_document_ids)
                job.progress = 0.8
            
            # Update metadata
            with self._lock:
                metadata = self.index_metadata[index_id]
                metadata.last_updated = time.time()
                metadata.status = IndexStatus.READY
                
                if new_document_ids:
                    metadata.document_count += len(new_document_ids)
                if remove_document_ids:
                    metadata.document_count -= len(remove_document_ids)
            
            job.progress = 0.9
            
            # Save updated index
            self.save_index(index_id)
            
            job.progress = 1.0
            job.status = "completed"
            
            self.logger.info(f"Index update completed for {index_id}")
            
        except Exception as e:
            self.logger.error(f"Index update failed for job {job_id}: {e}")
            job.status = "failed"
            job.error_message = str(e)
            
            # Restore index status
            if job_id in self.active_jobs:
                index_id = self.active_jobs[job_id].index_id
                if index_id in self.index_metadata:
                    self.index_metadata[index_id].status = IndexStatus.ERROR
        
        finally:
            self._cleanup_job(job_id)
    
    def _index_optimization_worker(self, job_id: str) -> None:
        """Worker thread for optimizing indexes."""
        try:
            job = self.active_jobs[job_id]
            index_id = job.index_id
            
            if index_id not in self.indexes:
                if not self.load_index(index_id):
                    raise RuntimeError(f"Failed to load index {index_id}")
            
            backend = self.indexes[index_id]
            job.status = "optimizing"
            job.progress = 0.1
            
            # Perform optimization
            optimization_result = backend.optimize_index()
            job.progress = 0.8
            
            # Update metadata
            with self._lock:
                metadata = self.index_metadata[index_id]
                metadata.last_optimized = time.time()
                metadata.status = IndexStatus.READY
                
                # Update performance stats
                if optimization_result.get("optimized", False):
                    metadata.performance_stats["last_optimization"] = optimization_result
            
            job.progress = 0.9
            
            # Save optimized index
            self.save_index(index_id)
            
            job.progress = 1.0
            job.status = "completed"
            
            self.logger.info(f"Index optimization completed for {index_id}")
            
        except Exception as e:
            self.logger.error(f"Index optimization failed for job {job_id}: {e}")
            job.status = "failed"
            job.error_message = str(e)
            
            # Restore index status
            if job_id in self.active_jobs:
                index_id = self.active_jobs[job_id].index_id
                if index_id in self.index_metadata:
                    self.index_metadata[index_id].status = IndexStatus.READY
        
        finally:
            self._cleanup_job(job_id)
    
    def _select_optimal_configuration(self, embeddings_shape: Tuple[int, int]) -> IndexConfiguration:
        """Select optimal index configuration based on data characteristics."""
        num_vectors, dimension = embeddings_shape
        
        # Select index type based on dataset size
        if num_vectors < 10000:
            index_type = "IndexFlatIP"
            parameters = {}
        elif num_vectors < 100000:
            index_type = "IndexIVFFlat"
            parameters = {"nlist": min(1024, num_vectors // 50)}
        elif num_vectors < 1000000:
            index_type = "IndexHNSWFlat"
            parameters = {"M": 32, "efConstruction": 200}
        else:
            index_type = "IndexIVFPQ"
            parameters = {"nlist": min(4096, num_vectors // 100), "m": 8, "bits": 8}
        
        return IndexConfiguration(
            index_type=index_type,
            dimension=dimension,
            parameters=parameters,
            backend_type="faiss",
            memory_limit_mb=min(2048, max(512, num_vectors * dimension * 4 // (1024 * 1024)))
        )
    
    def _find_ready_index(self, collection_id: str) -> Optional[str]:
        """Find ready index for collection."""
        for index_id, metadata in self.index_metadata.items():
            if (metadata.collection_id == collection_id and 
                metadata.status == IndexStatus.READY):
                return index_id
        return None
    
    def _should_rebuild_index(self, index_id: str, 
                             new_embeddings: Optional[np.ndarray]) -> bool:
        """Determine if index should be rebuilt instead of incrementally updated."""
        metadata = self.index_metadata[index_id]
        
        # Rebuild if adding significant amount of data
        if new_embeddings is not None:
            growth_ratio = len(new_embeddings) / max(metadata.document_count, 1)
            if growth_ratio > 0.5:  # Adding more than 50% new data
                return True
        
        # Rebuild if index is old
        age_hours = (time.time() - metadata.last_optimized) / 3600
        if age_hours > 24:  # Older than 24 hours
            return True
        
        return False
    
    def _get_index_path(self, index_id: str) -> str:
        """Get file path for index storage."""
        return os.path.join(self.storage_path, f"{index_id}.index")
    
    def _get_metadata_path(self, index_id: str) -> str:
        """Get file path for metadata storage."""
        return os.path.join(self.storage_path, f"{index_id}.metadata.json")
    
    def _save_metadata(self, index_id: str, metadata_path: str) -> None:
        """Save index metadata to file."""
        metadata = self.index_metadata[index_id]
        
        metadata_dict = {
            "index_id": metadata.index_id,
            "collection_id": metadata.collection_id,
            "version": metadata.version,
            "status": metadata.status.value,
            "configuration": {
                "index_type": metadata.configuration.index_type,
                "dimension": metadata.configuration.dimension,
                "parameters": metadata.configuration.parameters,
                "backend_type": metadata.configuration.backend_type,
                "enable_gpu": metadata.configuration.enable_gpu,
                "memory_limit_mb": metadata.configuration.memory_limit_mb,
                "optimization_enabled": metadata.configuration.optimization_enabled
            },
            "created_at": metadata.created_at,
            "last_updated": metadata.last_updated,
            "last_optimized": metadata.last_optimized,
            "document_count": metadata.document_count,
            "build_time_ms": metadata.build_time_ms,
            "memory_usage_mb": metadata.memory_usage_mb,
            "performance_stats": metadata.performance_stats
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _cleanup_job(self, job_id: str) -> None:
        """Clean up completed job."""
        with self._lock:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            if job_id in self.worker_threads:
                del self.worker_threads[job_id]
    
    def _stop_job(self, job_id: str) -> None:
        """Stop an active job."""
        with self._lock:
            if job_id in self.active_jobs:
                self.active_jobs[job_id].status = "stopped"
        
        self._cleanup_job(job_id)
    
    def _initialize_default_configurations(self) -> Dict[str, IndexConfiguration]:
        """Initialize default index configurations."""
        return {
            "small": IndexConfiguration(
                index_type="IndexFlatIP",
                dimension=512,
                backend_type="faiss"
            ),
            "medium": IndexConfiguration(
                index_type="IndexIVFFlat",
                dimension=512,
                parameters={"nlist": 1024},
                backend_type="faiss"
            ),
            "large": IndexConfiguration(
                index_type="IndexHNSWFlat",
                dimension=512,
                parameters={"M": 32, "efConstruction": 200},
                backend_type="faiss"
            )
        }
    
    def _initialize_optimization_thresholds(self) -> Dict[str, float]:
        """Initialize optimization thresholds."""
        return {
            "rebuild_growth_ratio": 0.5,
            "optimization_interval_hours": 24,
            "memory_usage_threshold": 0.8,
            "performance_degradation_threshold": 0.2
        }


__all__ = [
    "IndexStatus",
    "IndexConfiguration",
    "IndexMetadata", 
    "IndexUpdateJob",
    "DynamicIndexManager"
]