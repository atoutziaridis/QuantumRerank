"""
Advanced Search Engine for QuantumRerank Documentation.

This module provides sophisticated search capabilities including semantic search,
faceted search, auto-completion, and personalized recommendations for the
QuantumRerank documentation ecosystem.
"""

import re
import json
import time
import sqlite3
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import threading
from collections import defaultdict, Counter

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SearchScope(Enum):
    """Search scope definitions."""
    ALL = "all"
    DOCUMENTATION = "documentation"
    API = "api"
    TUTORIALS = "tutorials"
    EXAMPLES = "examples"
    TROUBLESHOOTING = "troubleshooting"
    RESEARCH = "research"


class RankingStrategy(Enum):
    """Ranking strategies for search results."""
    RELEVANCE = "relevance"
    POPULARITY = "popularity"
    RECENCY = "recency"
    HYBRID = "hybrid"
    PERSONALIZED = "personalized"


@dataclass
class SearchQuery:
    """Structured search query with advanced options."""
    query: str
    scope: SearchScope = SearchScope.ALL
    filters: Dict[str, Any] = field(default_factory=dict)
    ranking: RankingStrategy = RankingStrategy.HYBRID
    limit: int = 20
    offset: int = 0
    include_suggestions: bool = True
    include_facets: bool = True
    user_context: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Enhanced search result with rich metadata."""
    id: str
    title: str
    content: str
    url: str
    doc_type: str
    category: str
    tags: List[str]
    relevance_score: float
    popularity_score: float
    recency_score: float
    final_score: float
    snippet: str = ""
    highlighted_snippet: str = ""
    matched_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Comprehensive search response with all features."""
    query: SearchQuery
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    facets: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    auto_corrections: List[str] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    personalized_recommendations: List[SearchResult] = field(default_factory=list)
    search_id: str = ""


class DocumentIndex:
    """Advanced document indexing with multiple search capabilities."""
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize document index.
        
        Args:
            db_path: SQLite database path for persistent storage
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        
        # Create database schema
        self._create_schema()
        
        # In-memory indexes for fast access
        self.term_index = defaultdict(set)  # term -> set of doc_ids
        self.doc_vectors = {}  # doc_id -> embedding vector
        self.doc_metadata = {}  # doc_id -> metadata
        
        # Search analytics
        self.search_stats = {
            "total_searches": 0,
            "popular_terms": Counter(),
            "user_sessions": {}
        }
        
        self.logger = logger
        logger.info(f"Initialized DocumentIndex with database: {db_path}")
    
    def _create_schema(self) -> None:
        """Create database schema for document storage."""
        with self.lock:
            cursor = self.connection.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    url TEXT,
                    doc_type TEXT,
                    category TEXT,
                    tags TEXT,  -- JSON array
                    created_at REAL,
                    updated_at REAL,
                    view_count INTEGER DEFAULT 0,
                    rating REAL DEFAULT 0.0,
                    embedding BLOB  -- Serialized numpy array
                )
            """)
            
            # Search analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    user_id TEXT,
                    timestamp REAL,
                    results_count INTEGER,
                    clicked_result TEXT,
                    search_time_ms REAL
                )
            """)
            
            # Full-text search index
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    id, title, content, tags, 
                    content='documents',
                    content_rowid='rowid'
                )
            """)
            
            self.connection.commit()
    
    def add_document(self, doc_id: str, title: str, content: str,
                    url: str = "", doc_type: str = "", category: str = "",
                    tags: List[str] = None, embedding: np.ndarray = None) -> None:
        """Add document to index."""
        tags = tags or []
        
        with self.lock:
            cursor = self.connection.cursor()
            
            # Serialize embedding
            embedding_blob = None
            if embedding is not None:
                embedding_blob = embedding.tobytes()
                self.doc_vectors[doc_id] = embedding
            
            # Insert into documents table
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (id, title, content, url, doc_type, category, tags, created_at, updated_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, title, content, url, doc_type, category,
                json.dumps(tags), time.time(), time.time(), embedding_blob
            ))
            
            # Insert into FTS index
            cursor.execute("""
                INSERT OR REPLACE INTO documents_fts (id, title, content, tags)
                VALUES (?, ?, ?, ?)
            """, (doc_id, title, content, " ".join(tags)))
            
            self.connection.commit()
            
            # Update in-memory indexes
            self._update_term_index(doc_id, title + " " + content + " " + " ".join(tags))
            self.doc_metadata[doc_id] = {
                "title": title,
                "url": url,
                "doc_type": doc_type,
                "category": category,
                "tags": tags
            }
    
    def _update_term_index(self, doc_id: str, text: str) -> None:
        """Update in-memory term index."""
        # Simple tokenization (could be enhanced with proper NLP)
        terms = re.findall(r'\b\w+\b', text.lower())
        
        for term in terms:
            if len(term) > 2:  # Skip very short terms
                self.term_index[term].add(doc_id)
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM documents WHERE id = ?
            """, (doc_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "url": row[3],
                    "doc_type": row[4],
                    "category": row[5],
                    "tags": json.loads(row[6]),
                    "created_at": row[7],
                    "updated_at": row[8],
                    "view_count": row[9],
                    "rating": row[10]
                }
        return None
    
    def search_fulltext(self, query: str, limit: int = 50) -> List[Tuple[str, float]]:
        """Perform full-text search using SQLite FTS."""
        with self.lock:
            cursor = self.connection.cursor()
            
            # Use FTS5 MATCH for full-text search
            cursor.execute("""
                SELECT id, rank FROM documents_fts 
                WHERE documents_fts MATCH ? 
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            results = cursor.fetchall()
            
            # Convert rank to score (lower rank = higher relevance)
            scored_results = []
            for doc_id, rank in results:
                score = 1.0 / (1.0 + rank)  # Convert rank to 0-1 score
                scored_results.append((doc_id, score))
            
            return scored_results
    
    def search_semantic(self, query_vector: np.ndarray, limit: int = 50) -> List[Tuple[str, float]]:
        """Perform semantic search using vector similarity."""
        if not self.doc_vectors:
            return []
        
        similarities = []
        
        for doc_id, doc_vector in self.doc_vectors.items():
            # Cosine similarity
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )
            similarities.append((doc_id, float(similarity)))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:limit]
    
    def get_facets(self, doc_ids: Optional[Set[str]] = None) -> Dict[str, List[Tuple[str, int]]]:
        """Generate facets for search results."""
        with self.lock:
            cursor = self.connection.cursor()
            
            if doc_ids:
                # Facets for specific document set
                placeholders = ",".join("?" * len(doc_ids))
                cursor.execute(f"""
                    SELECT doc_type, category, tags FROM documents 
                    WHERE id IN ({placeholders})
                """, list(doc_ids))
            else:
                # Facets for all documents
                cursor.execute("""
                    SELECT doc_type, category, tags FROM documents
                """)
            
            rows = cursor.fetchall()
            
            facets = {
                "doc_type": Counter(),
                "category": Counter(),
                "tags": Counter()
            }
            
            for doc_type, category, tags_json in rows:
                if doc_type:
                    facets["doc_type"][doc_type] += 1
                if category:
                    facets["category"][category] += 1
                
                try:
                    tags = json.loads(tags_json)
                    for tag in tags:
                        facets["tags"][tag] += 1
                except:
                    pass
            
            # Convert to sorted lists
            result_facets = {}
            for facet_name, counter in facets.items():
                result_facets[facet_name] = counter.most_common(10)
            
            return result_facets
    
    def update_view_count(self, doc_id: str) -> None:
        """Update document view count."""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE documents SET view_count = view_count + 1 
                WHERE id = ?
            """, (doc_id,))
            self.connection.commit()
    
    def get_popular_documents(self, limit: int = 10) -> List[str]:
        """Get most popular documents by view count."""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id FROM documents 
                ORDER BY view_count DESC 
                LIMIT ?
            """, (limit,))
            
            return [row[0] for row in cursor.fetchall()]


class QueryProcessor:
    """Advanced query processing with spell correction and expansion."""
    
    def __init__(self):
        """Initialize query processor."""
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "will", "would", "could", "should", "may", "might"
        }
        
        # Common quantum/ML/IR terms for expansion
        self.domain_synonyms = {
            "quantum": ["quantum", "qubit", "quantum computing", "quantum mechanics"],
            "embedding": ["embedding", "vector", "representation", "encoding"],
            "similarity": ["similarity", "distance", "metric", "comparison"],
            "search": ["search", "retrieval", "query", "find"],
            "neural": ["neural", "deep learning", "machine learning", "AI"],
            "algorithm": ["algorithm", "method", "approach", "technique"]
        }
        
        self.logger = logger
        logger.info("Initialized QueryProcessor")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with normalization, expansion, and analysis."""
        original_query = query
        
        # Basic cleaning
        query = re.sub(r'[^\w\s]', ' ', query)  # Remove punctuation
        query = re.sub(r'\s+', ' ', query).strip()  # Normalize whitespace
        
        # Tokenize
        tokens = query.lower().split()
        
        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        # Extract key terms
        key_terms = filtered_tokens
        
        # Query expansion
        expanded_terms = self._expand_query_terms(key_terms)
        
        # Generate variations
        variations = self._generate_query_variations(original_query)
        
        return {
            "original": original_query,
            "cleaned": query,
            "tokens": tokens,
            "key_terms": key_terms,
            "expanded_terms": expanded_terms,
            "variations": variations,
            "query_type": self._classify_query_type(original_query)
        }
    
    def _expand_query_terms(self, terms: List[str]) -> List[str]:
        """Expand query terms with synonyms and related terms."""
        expanded = set(terms)
        
        for term in terms:
            # Check domain synonyms
            for key, synonyms in self.domain_synonyms.items():
                if term in key or key in term:
                    expanded.update(synonyms)
        
        return list(expanded)
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for better matching."""
        variations = [query]
        
        # Add singular/plural variations (simplified)
        words = query.split()
        for i, word in enumerate(words):
            if word.endswith('s') and len(word) > 3:
                # Try singular
                singular_words = words.copy()
                singular_words[i] = word[:-1]
                variations.append(" ".join(singular_words))
            elif not word.endswith('s'):
                # Try plural
                plural_words = words.copy()
                plural_words[i] = word + 's'
                variations.append(" ".join(plural_words))
        
        return list(set(variations))
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for specialized handling."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["how", "what", "why", "when", "where"]):
            return "question"
        elif any(word in query_lower for word in ["error", "problem", "issue", "trouble"]):
            return "troubleshooting"
        elif any(word in query_lower for word in ["tutorial", "guide", "example", "how to"]):
            return "tutorial"
        elif any(word in query_lower for word in ["api", "function", "method", "class"]):
            return "api"
        else:
            return "general"


class RankingEngine:
    """Advanced ranking engine with multiple strategies."""
    
    def __init__(self):
        """Initialize ranking engine."""
        self.logger = logger
        logger.info("Initialized RankingEngine")
    
    def rank_results(self, results: List[Tuple[str, float]], 
                    doc_index: DocumentIndex,
                    strategy: RankingStrategy = RankingStrategy.HYBRID,
                    user_context: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Rank search results using specified strategy.
        
        Args:
            results: List of (doc_id, relevance_score) tuples
            doc_index: Document index for metadata lookup
            strategy: Ranking strategy to use
            user_context: Optional user context for personalization
            
        Returns:
            List of ranked SearchResult objects
        """
        ranked_results = []
        
        for doc_id, relevance_score in results:
            doc = doc_index.get_document(doc_id)
            if not doc:
                continue
            
            # Calculate different scoring components
            popularity_score = self._calculate_popularity_score(doc)
            recency_score = self._calculate_recency_score(doc)
            
            # Calculate final score based on strategy
            if strategy == RankingStrategy.RELEVANCE:
                final_score = relevance_score
            elif strategy == RankingStrategy.POPULARITY:
                final_score = popularity_score
            elif strategy == RankingStrategy.RECENCY:
                final_score = recency_score
            elif strategy == RankingStrategy.HYBRID:
                final_score = (
                    relevance_score * 0.6 +
                    popularity_score * 0.3 +
                    recency_score * 0.1
                )
            elif strategy == RankingStrategy.PERSONALIZED:
                final_score = self._calculate_personalized_score(
                    relevance_score, popularity_score, recency_score,
                    doc, user_context
                )
            else:
                final_score = relevance_score
            
            # Create search result
            search_result = SearchResult(
                id=doc_id,
                title=doc["title"],
                content=doc["content"],
                url=doc["url"],
                doc_type=doc["doc_type"],
                category=doc["category"],
                tags=doc["tags"],
                relevance_score=relevance_score,
                popularity_score=popularity_score,
                recency_score=recency_score,
                final_score=final_score,
                metadata={"view_count": doc["view_count"], "rating": doc["rating"]}
            )
            
            ranked_results.append(search_result)
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return ranked_results
    
    def _calculate_popularity_score(self, doc: Dict[str, Any]) -> float:
        """Calculate popularity score based on views and ratings."""
        view_count = doc.get("view_count", 0)
        rating = doc.get("rating", 0.0)
        
        # Normalize view count (logarithmic scaling)
        view_score = np.log(view_count + 1) / np.log(1000)  # Normalize to ~1000 views
        view_score = min(1.0, view_score)
        
        # Normalize rating (assuming 0-5 scale)
        rating_score = rating / 5.0
        
        # Combine scores
        popularity_score = (view_score * 0.7 + rating_score * 0.3)
        return popularity_score
    
    def _calculate_recency_score(self, doc: Dict[str, Any]) -> float:
        """Calculate recency score based on document age."""
        updated_at = doc.get("updated_at", 0)
        current_time = time.time()
        
        # Age in days
        age_days = (current_time - updated_at) / (24 * 3600)
        
        # Exponential decay with 30-day half-life
        recency_score = np.exp(-age_days / 30)
        return recency_score
    
    def _calculate_personalized_score(self, relevance_score: float,
                                    popularity_score: float,
                                    recency_score: float,
                                    doc: Dict[str, Any],
                                    user_context: Optional[Dict[str, Any]]) -> float:
        """Calculate personalized score based on user context."""
        base_score = (
            relevance_score * 0.5 +
            popularity_score * 0.3 +
            recency_score * 0.2
        )
        
        if not user_context:
            return base_score
        
        # Adjust based on user preferences
        user_preferences = user_context.get("preferences", {})
        user_history = user_context.get("history", [])
        
        # Boost based on preferred document types
        preferred_types = user_preferences.get("doc_types", [])
        if doc["doc_type"] in preferred_types:
            base_score *= 1.2
        
        # Boost based on preferred categories
        preferred_categories = user_preferences.get("categories", [])
        if doc["category"] in preferred_categories:
            base_score *= 1.1
        
        # Boost based on user history
        if doc["id"] in user_history:
            base_score *= 0.9  # Slightly reduce score for already viewed docs
        
        return min(1.0, base_score)


class DocumentationSearchEngine:
    """
    Comprehensive search engine for QuantumRerank documentation.
    
    Combines full-text search, semantic search, faceted search,
    and personalized ranking for optimal search experience.
    """
    
    def __init__(self, db_path: str = ":memory:",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize documentation search engine.
        
        Args:
            db_path: SQLite database path
            embedding_model: Sentence transformer model for semantic search
        """
        self.doc_index = DocumentIndex(db_path)
        self.query_processor = QueryProcessor()
        self.ranking_engine = RankingEngine()
        
        # Initialize semantic search if available
        self.semantic_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer(embedding_model)
                logger.info(f"Initialized semantic search with {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
        
        # Search analytics
        self.search_analytics = defaultdict(int)
        self.query_suggestions = defaultdict(int)
        
        self.logger = logger
        logger.info("Initialized DocumentationSearchEngine")
    
    def index_document(self, doc_id: str, title: str, content: str,
                      url: str = "", doc_type: str = "", category: str = "",
                      tags: List[str] = None) -> None:
        """Index a document for search."""
        # Generate embedding if semantic model is available
        embedding = None
        if self.semantic_model:
            try:
                text_for_embedding = f"{title} {content}"
                embedding = self.semantic_model.encode([text_for_embedding])[0]
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {doc_id}: {e}")
        
        # Add to index
        self.doc_index.add_document(
            doc_id, title, content, url, doc_type, category, tags, embedding
        )
    
    def search(self, search_query: Union[str, SearchQuery]) -> SearchResponse:
        """
        Perform comprehensive search with all features.
        
        Args:
            search_query: Search query string or SearchQuery object
            
        Returns:
            SearchResponse with ranked results and metadata
        """
        start_time = time.time()
        
        # Convert string query to SearchQuery object
        if isinstance(search_query, str):
            search_query = SearchQuery(query=search_query)
        
        # Process query
        processed_query = self.query_processor.process_query(search_query.query)
        
        # Combine different search methods
        all_results = {}
        
        # 1. Full-text search
        fulltext_results = self.doc_index.search_fulltext(
            search_query.query, limit=search_query.limit * 2
        )
        for doc_id, score in fulltext_results:
            all_results[doc_id] = all_results.get(doc_id, 0) + score * 0.4
        
        # 2. Semantic search (if available)
        if self.semantic_model:
            try:
                query_embedding = self.semantic_model.encode([search_query.query])[0]
                semantic_results = self.doc_index.search_semantic(
                    query_embedding, limit=search_query.limit * 2
                )
                for doc_id, score in semantic_results:
                    all_results[doc_id] = all_results.get(doc_id, 0) + score * 0.6
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        # 3. Apply filters
        filtered_results = self._apply_filters(all_results, search_query.filters)
        
        # 4. Rank results
        result_tuples = list(filtered_results.items())
        ranked_results = self.ranking_engine.rank_results(
            result_tuples, self.doc_index, search_query.ranking, search_query.user_context
        )
        
        # 5. Generate snippets and highlights
        for result in ranked_results:
            result.snippet = self._generate_snippet(result, search_query.query)
            result.highlighted_snippet = self._highlight_snippet(result.snippet, search_query.query)
            result.matched_terms = self._find_matched_terms(result, processed_query["key_terms"])
        
        # 6. Paginate results
        start_idx = search_query.offset
        end_idx = start_idx + search_query.limit
        paginated_results = ranked_results[start_idx:end_idx]
        
        # 7. Generate facets
        facets = {}
        if search_query.include_facets:
            result_doc_ids = {result.id for result in ranked_results}
            facets = self.doc_index.get_facets(result_doc_ids)
        
        # 8. Generate suggestions
        suggestions = []
        if search_query.include_suggestions:
            suggestions = self._generate_suggestions(search_query.query, processed_query)
        
        # 9. Generate auto-corrections
        auto_corrections = self._generate_auto_corrections(search_query.query)
        
        # 10. Generate related queries
        related_queries = self._generate_related_queries(search_query.query)
        
        # 11. Generate personalized recommendations
        recommendations = []
        if search_query.user_context:
            recommendations = self._generate_recommendations(search_query.user_context)
        
        search_time = (time.time() - start_time) * 1000
        
        # Track analytics
        self._track_search_analytics(search_query, len(ranked_results), search_time)
        
        # Generate search ID for tracking
        search_id = f"search_{int(time.time() * 1000000)}"
        
        return SearchResponse(
            query=search_query,
            results=paginated_results,
            total_results=len(ranked_results),
            search_time_ms=search_time,
            facets=facets,
            suggestions=suggestions,
            auto_corrections=auto_corrections,
            related_queries=related_queries,
            personalized_recommendations=recommendations,
            search_id=search_id
        )
    
    def _apply_filters(self, results: Dict[str, float],
                      filters: Dict[str, Any]) -> Dict[str, float]:
        """Apply search filters to results."""
        if not filters:
            return results
        
        filtered_results = {}
        
        for doc_id, score in results.items():
            doc = self.doc_index.get_document(doc_id)
            if not doc:
                continue
            
            # Apply filters
            include_doc = True
            
            if "doc_type" in filters:
                allowed_types = filters["doc_type"]
                if isinstance(allowed_types, str):
                    allowed_types = [allowed_types]
                if doc["doc_type"] not in allowed_types:
                    include_doc = False
            
            if "category" in filters:
                allowed_categories = filters["category"]
                if isinstance(allowed_categories, str):
                    allowed_categories = [allowed_categories]
                if doc["category"] not in allowed_categories:
                    include_doc = False
            
            if "tags" in filters:
                required_tags = filters["tags"]
                if isinstance(required_tags, str):
                    required_tags = [required_tags]
                if not any(tag in doc["tags"] for tag in required_tags):
                    include_doc = False
            
            if include_doc:
                filtered_results[doc_id] = score
        
        return filtered_results
    
    def _generate_snippet(self, result: SearchResult, query: str,
                         max_length: int = 200) -> str:
        """Generate content snippet for search result."""
        content = result.content
        query_terms = query.lower().split()
        
        # Find the best sentence containing query terms
        sentences = re.split(r'[.!?]+', content)
        best_sentence_idx = 0
        best_score = 0
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence_idx = i
        
        # Create snippet around best sentence
        start_idx = max(0, best_sentence_idx - 1)
        end_idx = min(len(sentences), best_sentence_idx + 2)
        
        snippet = ". ".join(sentences[start_idx:end_idx]).strip()
        
        # Truncate if too long
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."
        
        return snippet
    
    def _highlight_snippet(self, snippet: str, query: str) -> str:
        """Highlight query terms in snippet."""
        query_terms = query.split()
        highlighted = snippet
        
        for term in query_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term}**", highlighted)
        
        return highlighted
    
    def _find_matched_terms(self, result: SearchResult, key_terms: List[str]) -> List[str]:
        """Find which query terms matched in the result."""
        content_lower = (result.title + " " + result.content).lower()
        matched = []
        
        for term in key_terms:
            if term.lower() in content_lower:
                matched.append(term)
        
        return matched
    
    def _generate_suggestions(self, query: str, processed_query: Dict[str, Any]) -> List[str]:
        """Generate search suggestions."""
        suggestions = []
        
        # Add expanded terms as suggestions
        for term in processed_query["expanded_terms"]:
            if term not in query.lower():
                suggestions.append(f"{query} {term}")
        
        # Add popular related queries
        popular_queries = [
            "getting started",
            "api reference",
            "tutorial",
            "examples",
            "troubleshooting"
        ]
        
        for popular in popular_queries:
            if popular not in query.lower() and len(suggestions) < 5:
                suggestions.append(f"{query} {popular}")
        
        return suggestions[:5]
    
    def _generate_auto_corrections(self, query: str) -> List[str]:
        """Generate auto-corrections for potential misspellings."""
        # Simplified spell checking (would be enhanced with proper spell checker)
        common_corrections = {
            "quantam": "quantum",
            "algoritm": "algorithm",
            "machien": "machine",
            "learing": "learning",
            "similary": "similarity",
            "embeding": "embedding"
        }
        
        corrections = []
        words = query.split()
        
        for i, word in enumerate(words):
            if word.lower() in common_corrections:
                corrected_words = words.copy()
                corrected_words[i] = common_corrections[word.lower()]
                corrections.append(" ".join(corrected_words))
        
        return corrections
    
    def _generate_related_queries(self, query: str) -> List[str]:
        """Generate related queries based on search patterns."""
        related = []
        
        # Query expansion based on domain knowledge
        if "quantum" in query.lower():
            related.extend([
                "quantum algorithms",
                "quantum computing tutorial",
                "quantum fidelity",
                "quantum circuits"
            ])
        
        if "embedding" in query.lower():
            related.extend([
                "vector embeddings",
                "similarity search",
                "semantic search",
                "embedding models"
            ])
        
        return related[:5]
    
    def _generate_recommendations(self, user_context: Dict[str, Any]) -> List[SearchResult]:
        """Generate personalized recommendations."""
        recommendations = []
        
        # Get popular documents in user's preferred categories
        preferred_categories = user_context.get("preferences", {}).get("categories", [])
        
        if preferred_categories:
            # Simple recommendation based on popularity in preferred categories
            popular_docs = self.doc_index.get_popular_documents(limit=10)
            
            for doc_id in popular_docs:
                doc = self.doc_index.get_document(doc_id)
                if doc and doc["category"] in preferred_categories:
                    recommendations.append(SearchResult(
                        id=doc_id,
                        title=doc["title"],
                        content=doc["content"],
                        url=doc["url"],
                        doc_type=doc["doc_type"],
                        category=doc["category"],
                        tags=doc["tags"],
                        relevance_score=0.8,
                        popularity_score=1.0,
                        recency_score=0.5,
                        final_score=0.8
                    ))
                
                if len(recommendations) >= 3:
                    break
        
        return recommendations
    
    def _track_search_analytics(self, query: SearchQuery, result_count: int,
                              search_time: float) -> None:
        """Track search analytics."""
        self.search_analytics["total_searches"] += 1
        self.search_analytics["total_results"] += result_count
        self.search_analytics["total_time"] += search_time
        
        # Track query terms
        terms = query.query.lower().split()
        for term in terms:
            self.query_suggestions[term] += 1
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics summary."""
        total_searches = self.search_analytics["total_searches"]
        
        if total_searches == 0:
            return {"total_searches": 0}
        
        avg_results = self.search_analytics["total_results"] / total_searches
        avg_time = self.search_analytics["total_time"] / total_searches
        
        popular_terms = dict(self.query_suggestions.most_common(10))
        
        return {
            "total_searches": total_searches,
            "avg_results_per_search": avg_results,
            "avg_search_time_ms": avg_time,
            "popular_search_terms": popular_terms
        }