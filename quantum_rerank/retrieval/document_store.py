"""
Document store for managing documents and their embeddings.

This module provides document management capabilities including ingestion,
preprocessing, embedding generation, and metadata handling.
"""

import logging
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import pickle

from ..core.embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata associated with a document."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
            "author": self.author,
            "title": self.title,
            "tags": self.tags,
            **self.custom_fields
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create metadata from dictionary."""
        # Extract known fields
        created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        source = data.get("source")
        author = data.get("author")
        title = data.get("title")
        tags = data.get("tags", [])
        
        # Everything else goes to custom_fields
        known_fields = {"created_at", "updated_at", "source", "author", "title", "tags"}
        custom_fields = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            created_at=created_at,
            updated_at=updated_at,
            source=source,
            author=author,
            title=title,
            tags=tags,
            custom_fields=custom_fields
        )


@dataclass
class Document:
    """
    Document representation with content, embedding, and metadata.
    """
    doc_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    
    def __post_init__(self):
        """Validate document after initialization."""
        if not self.doc_id:
            raise ValueError("Document ID cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")
    
    @classmethod
    def from_text(cls, 
                  text: str, 
                  doc_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> "Document":
        """
        Create document from text content.
        
        Args:
            text: Document content
            doc_id: Optional document ID (generated if not provided)
            metadata: Optional metadata dictionary
            
        Returns:
            Document instance
        """
        if doc_id is None:
            # Generate ID from content hash
            doc_id = hashlib.md5(text.encode()).hexdigest()[:16]
        
        meta = DocumentMetadata()
        if metadata:
            meta = DocumentMetadata.from_dict(metadata)
        
        return cls(doc_id=doc_id, content=text, metadata=meta)


class DocumentStore:
    """
    Document store for managing documents, embeddings, and metadata.
    
    Provides functionality for document ingestion, preprocessing,
    embedding generation, and batch operations.
    """
    
    def __init__(self, 
                 embedding_processor: Optional[EmbeddingProcessor] = None,
                 max_content_length: int = 10000,
                 enable_deduplication: bool = True):
        """
        Initialize document store.
        
        Args:
            embedding_processor: Processor for generating embeddings
            max_content_length: Maximum allowed document content length
            enable_deduplication: Whether to check for duplicate documents
        """
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        self.max_content_length = max_content_length
        self.enable_deduplication = enable_deduplication
        
        # Storage
        self.documents: Dict[str, Document] = {}
        self.content_hashes: Dict[str, str] = {}  # For deduplication
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_embeddings_generated": 0,
            "duplicate_documents_skipped": 0,
            "avg_document_length": 0.0
        }
        
        logger.info("Document store initialized")
    
    def add_document(self, 
                    document: Document,
                    generate_embedding: bool = True,
                    update_if_exists: bool = False) -> bool:
        """
        Add a single document to the store.
        
        Args:
            document: Document to add
            generate_embedding: Whether to generate embedding immediately
            update_if_exists: Whether to update existing documents
            
        Returns:
            True if document was added/updated, False if skipped
        """
        # Check document length
        if len(document.content) > self.max_content_length:
            logger.warning(f"Document {document.doc_id} exceeds max length "
                         f"({len(document.content)} > {self.max_content_length})")
            document.content = document.content[:self.max_content_length]
        
        # Check for duplicates
        if self.enable_deduplication:
            content_hash = hashlib.md5(document.content.encode()).hexdigest()
            
            if content_hash in self.content_hashes:
                existing_id = self.content_hashes[content_hash]
                if existing_id != document.doc_id:
                    logger.info(f"Duplicate content detected. Document {document.doc_id} "
                              f"has same content as {existing_id}")
                    self.stats["duplicate_documents_skipped"] += 1
                    return False
        
        # Check if document exists
        if document.doc_id in self.documents and not update_if_exists:
            logger.warning(f"Document {document.doc_id} already exists. "
                         "Set update_if_exists=True to update.")
            return False
        
        # Generate embedding if requested
        if generate_embedding and document.embedding is None:
            embedding = self.embedding_processor.encode_texts([document.content])[0]
            document.embedding = embedding.tolist()
            self.stats["total_embeddings_generated"] += 1
        
        # Update metadata timestamp
        if document.doc_id in self.documents:
            document.metadata.updated_at = datetime.now()
        
        # Store document
        self.documents[document.doc_id] = document
        if self.enable_deduplication:
            self.content_hashes[content_hash] = document.doc_id
        
        # Update statistics
        self._update_stats()
        
        logger.debug(f"Added document {document.doc_id} "
                    f"(length: {len(document.content)} chars)")
        
        return True
    
    def add_documents(self,
                     documents: List[Document],
                     generate_embeddings: bool = True,
                     batch_size: int = 32,
                     show_progress: bool = True) -> int:
        """
        Add multiple documents in batch.
        
        Args:
            documents: List of documents to add
            generate_embeddings: Whether to generate embeddings
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress
            
        Returns:
            Number of documents successfully added
        """
        added = 0
        
        # Process in batches for embedding generation
        if generate_embeddings:
            docs_without_embeddings = [
                doc for doc in documents if doc.embedding is None
            ]
            
            if docs_without_embeddings:
                logger.info(f"Generating embeddings for {len(docs_without_embeddings)} documents")
                
                for i in range(0, len(docs_without_embeddings), batch_size):
                    batch = docs_without_embeddings[i:i + batch_size]
                    texts = [doc.content for doc in batch]
                    
                    # Generate embeddings
                    embeddings = self.embedding_processor.encode_texts(texts)
                    
                    # Assign embeddings
                    for doc, embedding in zip(batch, embeddings):
                        doc.embedding = embedding.tolist()
                        self.stats["total_embeddings_generated"] += 1
                    
                    if show_progress and (i + batch_size) % 100 == 0:
                        logger.info(f"Processed {i + batch_size}/{len(docs_without_embeddings)} documents")
        
        # Add documents
        for document in documents:
            if self.add_document(document, generate_embedding=False):
                added += 1
        
        logger.info(f"Added {added} documents to store "
                   f"(total: {self.stats['total_documents']})")
        
        return added
    
    def add_texts(self,
                 texts: List[str],
                 metadatas: Optional[List[Dict[str, Any]]] = None,
                 doc_ids: Optional[List[str]] = None) -> List[str]:
        """
        Add texts as documents.
        
        Args:
            texts: List of text contents
            metadatas: Optional metadata for each text
            doc_ids: Optional document IDs
            
        Returns:
            List of document IDs that were added
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        if doc_ids is None:
            doc_ids = [None] * len(texts)
        
        if len(texts) != len(metadatas) or len(texts) != len(doc_ids):
            raise ValueError("texts, metadatas, and doc_ids must have same length")
        
        documents = []
        for text, metadata, doc_id in zip(texts, metadatas, doc_ids):
            doc = Document.from_text(text, doc_id=doc_id, metadata=metadata)
            documents.append(doc)
        
        self.add_documents(documents)
        
        return [doc.doc_id for doc in documents]
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)
    
    def get_documents(self, 
                     doc_ids: Optional[List[str]] = None,
                     filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve multiple documents.
        
        Args:
            doc_ids: Specific document IDs to retrieve
            filter_dict: Metadata filters to apply
            
        Returns:
            List of documents matching criteria
        """
        if doc_ids is not None:
            documents = [self.documents[doc_id] for doc_id in doc_ids 
                        if doc_id in self.documents]
        else:
            documents = list(self.documents.values())
        
        # Apply filters
        if filter_dict:
            filtered = []
            for doc in documents:
                match = all(
                    doc.metadata.to_dict().get(key) == value
                    for key, value in filter_dict.items()
                )
                if match:
                    filtered.append(doc)
            documents = filtered
        
        return documents
    
    def update_document(self,
                       doc_id: str,
                       content: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       regenerate_embedding: bool = True) -> bool:
        """
        Update existing document.
        
        Args:
            doc_id: Document ID to update
            content: New content (if provided)
            metadata: Metadata updates
            regenerate_embedding: Whether to regenerate embedding if content changed
            
        Returns:
            True if document was updated
        """
        if doc_id not in self.documents:
            logger.warning(f"Document {doc_id} not found")
            return False
        
        doc = self.documents[doc_id]
        updated = False
        
        # Update content
        if content is not None and content != doc.content:
            doc.content = content
            doc.metadata.updated_at = datetime.now()
            updated = True
            
            # Regenerate embedding if requested
            if regenerate_embedding:
                embedding = self.embedding_processor.encode_texts([content])[0]
                doc.embedding = embedding.tolist()
                self.stats["total_embeddings_generated"] += 1
        
        # Update metadata
        if metadata is not None:
            for key, value in metadata.items():
                if hasattr(doc.metadata, key):
                    setattr(doc.metadata, key, value)
                else:
                    doc.metadata.custom_fields[key] = value
            doc.metadata.updated_at = datetime.now()
            updated = True
        
        if updated:
            logger.info(f"Updated document {doc_id}")
        
        return updated
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove document from store.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if document was removed
        """
        if doc_id not in self.documents:
            return False
        
        doc = self.documents[doc_id]
        
        # Remove from content hashes
        if self.enable_deduplication:
            content_hash = hashlib.md5(doc.content.encode()).hexdigest()
            if content_hash in self.content_hashes:
                del self.content_hashes[content_hash]
        
        # Remove document
        del self.documents[doc_id]
        
        # Update stats
        self._update_stats()
        
        logger.info(f"Removed document {doc_id}")
        return True
    
    def remove_documents(self, doc_ids: List[str]) -> int:
        """Remove multiple documents."""
        removed = 0
        for doc_id in doc_ids:
            if self.remove_document(doc_id):
                removed += 1
        return removed
    
    def clear(self):
        """Clear all documents from store."""
        self.documents.clear()
        self.content_hashes.clear()
        self.stats = {
            "total_documents": 0,
            "total_embeddings_generated": 0,
            "duplicate_documents_skipped": 0,
            "avg_document_length": 0.0
        }
        logger.info("Document store cleared")
    
    def save(self, path: str):
        """Save document store to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "documents": self.documents,
            "content_hashes": self.content_hashes,
            "stats": self.stats,
            "config": {
                "max_content_length": self.max_content_length,
                "enable_deduplication": self.enable_deduplication
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved document store to {path} "
                   f"({len(self.documents)} documents)")
    
    def load(self, path: str):
        """Load document store from disk."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.content_hashes = data["content_hashes"]
        self.stats = data["stats"]
        self.max_content_length = data["config"]["max_content_length"]
        self.enable_deduplication = data["config"]["enable_deduplication"]
        
        logger.info(f"Loaded document store from {path} "
                   f"({len(self.documents)} documents)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics."""
        return {
            **self.stats,
            "embeddings_coverage": self._get_embedding_coverage(),
            "metadata_fields": self._get_metadata_fields()
        }
    
    def _update_stats(self):
        """Update internal statistics."""
        self.stats["total_documents"] = len(self.documents)
        
        if self.documents:
            total_length = sum(len(doc.content) for doc in self.documents.values())
            self.stats["avg_document_length"] = total_length / len(self.documents)
        else:
            self.stats["avg_document_length"] = 0.0
    
    def _get_embedding_coverage(self) -> float:
        """Calculate percentage of documents with embeddings."""
        if not self.documents:
            return 0.0
        
        with_embeddings = sum(
            1 for doc in self.documents.values() 
            if doc.embedding is not None
        )
        
        return (with_embeddings / len(self.documents)) * 100
    
    def _get_metadata_fields(self) -> List[str]:
        """Get all unique metadata fields across documents."""
        fields = set()
        
        for doc in self.documents.values():
            fields.update(doc.metadata.to_dict().keys())
        
        return sorted(list(fields))