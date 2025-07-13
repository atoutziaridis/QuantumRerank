"""
Comprehensive Knowledge Management System for QuantumRerank.

This module provides intelligent documentation search, contextual help,
interactive tutorials, and comprehensive knowledge base functionality
for the QuantumRerank quantum-enhanced information retrieval system.
"""

import re
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import sqlite3
import threading

# For semantic search capabilities
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DocumentationType(Enum):
    """Types of documentation content."""
    TECHNICAL = "technical"
    API = "api"
    TUTORIAL = "tutorial"
    TROUBLESHOOTING = "troubleshooting"
    EXAMPLE = "example"
    BEST_PRACTICE = "best_practice"
    FAQ = "faq"
    RESEARCH = "research"


class SearchMode(Enum):
    """Search modes for documentation."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    FUZZY = "fuzzy"
    HYBRID = "hybrid"


@dataclass
class DocumentationItem:
    """Individual documentation item."""
    id: str
    title: str
    content: str
    doc_type: DocumentationType
    tags: List[str] = field(default_factory=list)
    category: str = ""
    difficulty_level: str = "intermediate"  # beginner, intermediate, advanced
    last_updated: float = field(default_factory=time.time)
    view_count: int = 0
    rating: float = 0.0
    related_items: List[str] = field(default_factory=list)
    code_examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result with relevance scoring."""
    item: DocumentationItem
    relevance_score: float
    matched_sections: List[str] = field(default_factory=list)
    highlighted_content: str = ""
    search_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResults:
    """Collection of search results with metadata."""
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    search_query: str
    search_mode: SearchMode
    facets: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ContextualHelp:
    """Contextual help information."""
    component: str
    operation: str
    description: str
    usage_examples: List[Dict[str, Any]] = field(default_factory=list)
    common_issues: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    quick_actions: List[Dict[str, str]] = field(default_factory=list)


class DocumentationSearchEngine:
    """Advanced search engine for documentation with semantic capabilities."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize documentation search engine.
        
        Args:
            embedding_model: Sentence transformer model for semantic search
        """
        self.embedding_model_name = embedding_model
        self.semantic_model = None
        self.document_index = {}
        self.embedding_cache = {}
        self.search_analytics = {}
        
        # Initialize semantic search if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer(embedding_model)
                logger.info(f"Initialized semantic search with {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
        
        self.logger = logger
        logger.info("Initialized DocumentationSearchEngine")
    
    def index_document(self, doc: DocumentationItem) -> None:
        """Index a documentation item for search."""
        self.document_index[doc.id] = doc
        
        # Generate embeddings for semantic search
        if self.semantic_model:
            content_text = f"{doc.title} {doc.content} {' '.join(doc.tags)}"
            try:
                embedding = self.semantic_model.encode([content_text])[0]
                self.embedding_cache[doc.id] = embedding
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {doc.id}: {e}")
    
    def search(self, query: str, mode: SearchMode = SearchMode.HYBRID,
              filters: Optional[Dict[str, Any]] = None,
              limit: int = 20) -> SearchResults:
        """
        Search documentation with multiple modes.
        
        Args:
            query: Search query
            mode: Search mode to use
            filters: Optional filters (doc_type, category, difficulty_level)
            limit: Maximum number of results
            
        Returns:
            SearchResults with ranked results
        """
        start_time = time.time()
        
        # Apply filters first
        candidate_docs = list(self.document_index.values())
        if filters:
            candidate_docs = self._apply_filters(candidate_docs, filters)
        
        # Perform search based on mode
        if mode == SearchMode.SEMANTIC and self.semantic_model:
            results = self._semantic_search(query, candidate_docs, limit)
        elif mode == SearchMode.KEYWORD:
            results = self._keyword_search(query, candidate_docs, limit)
        elif mode == SearchMode.FUZZY:
            results = self._fuzzy_search(query, candidate_docs, limit)
        elif mode == SearchMode.HYBRID:
            results = self._hybrid_search(query, candidate_docs, limit)
        else:
            # Fallback to keyword search
            results = self._keyword_search(query, candidate_docs, limit)
        
        search_time = (time.time() - start_time) * 1000
        
        # Generate facets and suggestions
        facets = self._generate_facets(candidate_docs)
        suggestions = self._generate_suggestions(query, results)
        
        # Track search analytics
        self._track_search(query, mode, len(results), search_time)
        
        return SearchResults(
            results=results,
            total_results=len(results),
            search_time_ms=search_time,
            search_query=query,
            search_mode=mode,
            facets=facets,
            suggestions=suggestions
        )
    
    def _apply_filters(self, docs: List[DocumentationItem],
                      filters: Dict[str, Any]) -> List[DocumentationItem]:
        """Apply search filters to documents."""
        filtered_docs = docs
        
        if 'doc_type' in filters:
            doc_types = filters['doc_type'] if isinstance(filters['doc_type'], list) else [filters['doc_type']]
            filtered_docs = [doc for doc in filtered_docs if doc.doc_type.value in doc_types]
        
        if 'category' in filters:
            categories = filters['category'] if isinstance(filters['category'], list) else [filters['category']]
            filtered_docs = [doc for doc in filtered_docs if doc.category in categories]
        
        if 'difficulty_level' in filters:
            levels = filters['difficulty_level'] if isinstance(filters['difficulty_level'], list) else [filters['difficulty_level']]
            filtered_docs = [doc for doc in filtered_docs if doc.difficulty_level in levels]
        
        if 'tags' in filters:
            required_tags = filters['tags'] if isinstance(filters['tags'], list) else [filters['tags']]
            filtered_docs = [doc for doc in filtered_docs if any(tag in doc.tags for tag in required_tags)]
        
        return filtered_docs
    
    def _semantic_search(self, query: str, docs: List[DocumentationItem],
                        limit: int) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        if not self.semantic_model:
            return self._keyword_search(query, docs, limit)
        
        try:
            query_embedding = self.semantic_model.encode([query])[0]
            scores = []
            
            for doc in docs:
                if doc.id in self.embedding_cache:
                    doc_embedding = self.embedding_cache[doc.id]
                    # Cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    scores.append((doc, similarity))
            
            # Sort by similarity score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for doc, score in scores[:limit]:
                highlighted_content = self._highlight_content(doc.content, query)
                results.append(SearchResult(
                    item=doc,
                    relevance_score=score,
                    highlighted_content=highlighted_content,
                    search_context={"method": "semantic", "embedding_model": self.embedding_model_name}
                ))
            
            return results
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return self._keyword_search(query, docs, limit)
    
    def _keyword_search(self, query: str, docs: List[DocumentationItem],
                       limit: int) -> List[SearchResult]:
        """Perform keyword-based search."""
        query_terms = query.lower().split()
        scores = []
        
        for doc in docs:
            score = 0.0
            matched_sections = []
            
            # Search in title (higher weight)
            title_matches = sum(1 for term in query_terms if term in doc.title.lower())
            score += title_matches * 3.0
            if title_matches > 0:
                matched_sections.append("title")
            
            # Search in content
            content_lower = doc.content.lower()
            content_matches = sum(1 for term in query_terms if term in content_lower)
            score += content_matches * 1.0
            if content_matches > 0:
                matched_sections.append("content")
            
            # Search in tags (medium weight)
            tag_matches = sum(1 for term in query_terms for tag in doc.tags if term in tag.lower())
            score += tag_matches * 2.0
            if tag_matches > 0:
                matched_sections.append("tags")
            
            # Boost score based on document type
            if doc.doc_type in [DocumentationType.TUTORIAL, DocumentationType.FAQ]:
                score *= 1.2
            
            if score > 0:
                scores.append((doc, score, matched_sections))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score, sections in scores[:limit]:
            # Normalize score to 0-1 range
            normalized_score = min(1.0, score / (len(query_terms) * 3.0))
            highlighted_content = self._highlight_content(doc.content, query)
            
            results.append(SearchResult(
                item=doc,
                relevance_score=normalized_score,
                matched_sections=sections,
                highlighted_content=highlighted_content,
                search_context={"method": "keyword"}
            ))
        
        return results
    
    def _fuzzy_search(self, query: str, docs: List[DocumentationItem],
                     limit: int) -> List[SearchResult]:
        """Perform fuzzy search with edit distance."""
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        query_lower = query.lower()
        scores = []
        
        for doc in docs:
            # Check fuzzy matches in title and content
            title_words = doc.title.lower().split()
            content_words = doc.content.lower().split()[:100]  # Limit for performance
            
            min_distance = float('inf')
            for word in title_words + content_words:
                if len(word) >= 3:  # Only check words with 3+ characters
                    distance = levenshtein_distance(query_lower, word)
                    min_distance = min(min_distance, distance)
            
            if min_distance < len(query) * 0.5:  # Allow 50% character differences
                similarity = 1.0 - (min_distance / len(query))
                scores.append((doc, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scores[:limit]:
            highlighted_content = self._highlight_content(doc.content, query)
            results.append(SearchResult(
                item=doc,
                relevance_score=score,
                highlighted_content=highlighted_content,
                search_context={"method": "fuzzy"}
            ))
        
        return results
    
    def _hybrid_search(self, query: str, docs: List[DocumentationItem],
                      limit: int) -> List[SearchResult]:
        """Combine semantic, keyword, and fuzzy search results."""
        # Get results from different methods
        semantic_results = self._semantic_search(query, docs, limit * 2) if self.semantic_model else []
        keyword_results = self._keyword_search(query, docs, limit * 2)
        fuzzy_results = self._fuzzy_search(query, docs, limit) if len(query) > 3 else []
        
        # Combine and deduplicate results
        combined_results = {}
        
        # Weight semantic results highest
        for result in semantic_results:
            doc_id = result.item.id
            if doc_id not in combined_results:
                result.relevance_score *= 0.6  # 60% weight for semantic
                combined_results[doc_id] = result
            else:
                combined_results[doc_id].relevance_score += result.relevance_score * 0.6
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result.item.id
            if doc_id not in combined_results:
                result.relevance_score *= 0.3  # 30% weight for keyword
                combined_results[doc_id] = result
            else:
                combined_results[doc_id].relevance_score += result.relevance_score * 0.3
        
        # Add fuzzy results
        for result in fuzzy_results:
            doc_id = result.item.id
            if doc_id not in combined_results:
                result.relevance_score *= 0.1  # 10% weight for fuzzy
                combined_results[doc_id] = result
            else:
                combined_results[doc_id].relevance_score += result.relevance_score * 0.1
        
        # Sort by combined score and return top results
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update search context
        for result in final_results:
            result.search_context["method"] = "hybrid"
        
        return final_results[:limit]
    
    def _highlight_content(self, content: str, query: str, max_length: int = 300) -> str:
        """Highlight search terms in content and create snippet."""
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        # Find the best snippet containing query terms
        best_start = 0
        best_matches = 0
        
        for i in range(0, len(content) - max_length, 50):
            snippet = content_lower[i:i+max_length]
            matches = sum(1 for term in query_terms if term in snippet)
            if matches > best_matches:
                best_matches = matches
                best_start = i
        
        snippet = content[best_start:best_start+max_length]
        
        # Highlight terms (simplified HTML-like highlighting)
        for term in query_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            snippet = pattern.sub(f"**{term}**", snippet)
        
        if best_start > 0:
            snippet = "..." + snippet
        if best_start + max_length < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def _generate_facets(self, docs: List[DocumentationItem]) -> Dict[str, List[Tuple[str, int]]]:
        """Generate facets for search results."""
        facets = {
            "doc_type": {},
            "category": {},
            "difficulty_level": {},
            "tags": {}
        }
        
        for doc in docs:
            # Document type facet
            doc_type = doc.doc_type.value
            facets["doc_type"][doc_type] = facets["doc_type"].get(doc_type, 0) + 1
            
            # Category facet
            if doc.category:
                facets["category"][doc.category] = facets["category"].get(doc.category, 0) + 1
            
            # Difficulty level facet
            level = doc.difficulty_level
            facets["difficulty_level"][level] = facets["difficulty_level"].get(level, 0) + 1
            
            # Tags facet (top 10 most common)
            for tag in doc.tags:
                facets["tags"][tag] = facets["tags"].get(tag, 0) + 1
        
        # Convert to sorted lists and limit tag facets
        result_facets = {}
        for facet_name, facet_counts in facets.items():
            sorted_facets = sorted(facet_counts.items(), key=lambda x: x[1], reverse=True)
            if facet_name == "tags":
                sorted_facets = sorted_facets[:10]  # Limit tags to top 10
            result_facets[facet_name] = sorted_facets
        
        return result_facets
    
    def _generate_suggestions(self, query: str, results: List[SearchResult]) -> List[str]:
        """Generate search suggestions based on query and results."""
        suggestions = []
        
        # Suggest related terms from top results
        if results:
            top_tags = set()
            for result in results[:5]:
                top_tags.update(result.item.tags)
            
            # Suggest tag-based queries
            for tag in list(top_tags)[:3]:
                if tag.lower() not in query.lower():
                    suggestions.append(f"{query} {tag}")
        
        # Suggest common documentation categories
        common_categories = [
            "getting started", "tutorial", "api reference", "troubleshooting",
            "examples", "best practices", "quantum algorithms", "performance"
        ]
        
        for category in common_categories:
            if category not in query.lower() and len(suggestions) < 5:
                suggestions.append(f"{query} {category}")
        
        return suggestions[:5]
    
    def _track_search(self, query: str, mode: SearchMode, result_count: int, search_time: float) -> None:
        """Track search analytics."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash not in self.search_analytics:
            self.search_analytics[query_hash] = {
                "query": query,
                "search_count": 0,
                "total_results": 0,
                "avg_search_time": 0.0,
                "modes_used": set()
            }
        
        analytics = self.search_analytics[query_hash]
        analytics["search_count"] += 1
        analytics["total_results"] += result_count
        analytics["avg_search_time"] = (
            (analytics["avg_search_time"] * (analytics["search_count"] - 1) + search_time) /
            analytics["search_count"]
        )
        analytics["modes_used"].add(mode.value)
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics summary."""
        if not self.search_analytics:
            return {"total_searches": 0}
        
        total_searches = sum(data["search_count"] for data in self.search_analytics.values())
        avg_results = np.mean([data["total_results"] / data["search_count"] for data in self.search_analytics.values()])
        avg_time = np.mean([data["avg_search_time"] for data in self.search_analytics.values()])
        
        most_searched = sorted(
            self.search_analytics.items(),
            key=lambda x: x[1]["search_count"],
            reverse=True
        )[:10]
        
        return {
            "total_searches": total_searches,
            "unique_queries": len(self.search_analytics),
            "avg_results_per_search": avg_results,
            "avg_search_time_ms": avg_time,
            "most_searched_queries": [(data["query"], data["search_count"]) for _, data in most_searched]
        }


class InteractiveCodeExamples:
    """Interactive code examples with execution capabilities."""
    
    def __init__(self):
        """Initialize interactive code examples system."""
        self.examples_db = {}
        self.execution_history = {}
        self.logger = logger
        
        logger.info("Initialized InteractiveCodeExamples")
    
    def add_example(self, example_id: str, title: str, code: str,
                   description: str, language: str = "python",
                   dependencies: List[str] = None,
                   expected_output: str = None) -> None:
        """Add a new interactive code example."""
        self.examples_db[example_id] = {
            "title": title,
            "code": code,
            "description": description,
            "language": language,
            "dependencies": dependencies or [],
            "expected_output": expected_output,
            "created_at": time.time(),
            "execution_count": 0,
            "success_rate": 0.0
        }
    
    def get_examples_for_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get interactive examples for a specific topic."""
        matching_examples = []
        
        for example_id, example in self.examples_db.items():
            if (topic.lower() in example["title"].lower() or
                topic.lower() in example["description"].lower()):
                matching_examples.append({
                    "id": example_id,
                    **example
                })
        
        return matching_examples
    
    def search_examples(self, query: str) -> List[Dict[str, Any]]:
        """Search for code examples matching query."""
        query_lower = query.lower()
        matching_examples = []
        
        for example_id, example in self.examples_db.items():
            score = 0
            
            # Search in title
            if query_lower in example["title"].lower():
                score += 3
            
            # Search in description
            if query_lower in example["description"].lower():
                score += 2
            
            # Search in code
            if query_lower in example["code"].lower():
                score += 1
            
            if score > 0:
                matching_examples.append({
                    "id": example_id,
                    "score": score,
                    **example
                })
        
        # Sort by score
        matching_examples.sort(key=lambda x: x["score"], reverse=True)
        return matching_examples
    
    def execute_example(self, example_id: str, user_modifications: str = None) -> Dict[str, Any]:
        """Execute code example (simulation - would integrate with safe execution environment)."""
        if example_id not in self.examples_db:
            return {"success": False, "error": "Example not found"}
        
        example = self.examples_db[example_id]
        code_to_execute = user_modifications or example["code"]
        
        # Simulate code execution (in real implementation would use sandboxed environment)
        execution_result = {
            "success": True,
            "output": example.get("expected_output", "Example executed successfully"),
            "execution_time": 0.1,
            "memory_used": "10MB",
            "warnings": [],
            "modified_code": code_to_execute != example["code"]
        }
        
        # Track execution statistics
        example["execution_count"] += 1
        
        # Store in execution history
        execution_record = {
            "example_id": example_id,
            "timestamp": time.time(),
            "success": execution_result["success"],
            "modified": execution_result["modified_code"]
        }
        
        if example_id not in self.execution_history:
            self.execution_history[example_id] = []
        self.execution_history[example_id].append(execution_record)
        
        return execution_result


class FAQSystem:
    """Frequently Asked Questions system with smart matching."""
    
    def __init__(self):
        """Initialize FAQ system."""
        self.faq_database = {}
        self.question_embeddings = {}
        self.semantic_model = None
        
        # Initialize semantic matching if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to load FAQ semantic model: {e}")
        
        self.logger = logger
        logger.info("Initialized FAQSystem")
    
    def add_faq(self, faq_id: str, question: str, answer: str,
               category: str = "", tags: List[str] = None) -> None:
        """Add a new FAQ item."""
        self.faq_database[faq_id] = {
            "question": question,
            "answer": answer,
            "category": category,
            "tags": tags or [],
            "created_at": time.time(),
            "view_count": 0,
            "helpful_votes": 0,
            "total_votes": 0
        }
        
        # Generate embedding for semantic matching
        if self.semantic_model:
            try:
                embedding = self.semantic_model.encode([question])[0]
                self.question_embeddings[faq_id] = embedding
            except Exception as e:
                logger.warning(f"Failed to generate FAQ embedding for {faq_id}: {e}")
    
    def find_matching_faqs(self, user_question: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find FAQs matching user question."""
        if self.semantic_model and self.question_embeddings:
            return self._semantic_faq_search(user_question, limit)
        else:
            return self._keyword_faq_search(user_question, limit)
    
    def _semantic_faq_search(self, user_question: str, limit: int) -> List[Dict[str, Any]]:
        """Use semantic similarity to find matching FAQs."""
        try:
            question_embedding = self.semantic_model.encode([user_question])[0]
            similarities = []
            
            for faq_id, faq_embedding in self.question_embeddings.items():
                similarity = np.dot(question_embedding, faq_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(faq_embedding)
                )
                similarities.append((faq_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for faq_id, similarity in similarities[:limit]:
                if similarity > 0.3:  # Minimum similarity threshold
                    faq = self.faq_database[faq_id]
                    results.append({
                        "id": faq_id,
                        "similarity": similarity,
                        **faq
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"Semantic FAQ search failed: {e}")
            return self._keyword_faq_search(user_question, limit)
    
    def _keyword_faq_search(self, user_question: str, limit: int) -> List[Dict[str, Any]]:
        """Use keyword matching to find FAQs."""
        user_words = user_question.lower().split()
        matches = []
        
        for faq_id, faq in self.faq_database.items():
            score = 0
            question_words = faq["question"].lower().split()
            
            # Count matching words
            matching_words = set(user_words) & set(question_words)
            score += len(matching_words) * 2
            
            # Check for partial word matches
            for user_word in user_words:
                for question_word in question_words:
                    if len(user_word) > 3 and user_word in question_word:
                        score += 1
            
            # Check tags
            for tag in faq["tags"]:
                if any(word in tag.lower() for word in user_words):
                    score += 1
            
            if score > 0:
                matches.append((faq_id, score))
        
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for faq_id, score in matches[:limit]:
            faq = self.faq_database[faq_id]
            results.append({
                "id": faq_id,
                "score": score,
                **faq
            })
        
        return results
    
    def get_common_issues(self, component: str) -> List[str]:
        """Get common issues for a specific component."""
        component_faqs = []
        
        for faq in self.faq_database.values():
            if (component.lower() in faq["question"].lower() or
                component.lower() in faq["category"].lower() or
                any(component.lower() in tag.lower() for tag in faq["tags"])):
                component_faqs.append(faq["question"])
        
        return component_faqs[:10]  # Return top 10 most relevant


class BestPracticesDatabase:
    """Database of best practices and patterns."""
    
    def __init__(self):
        """Initialize best practices database."""
        self.practices_db = {}
        self.logger = logger
        
        # Add default best practices
        self._initialize_default_practices()
        
        logger.info("Initialized BestPracticesDatabase")
    
    def _initialize_default_practices(self) -> None:
        """Initialize with default best practices."""
        default_practices = {
            "quantum_circuits": [
                "Keep circuit depth minimal for NISQ devices",
                "Use parameter reuse to reduce optimization complexity",
                "Implement error mitigation techniques",
                "Validate quantum advantages over classical methods"
            ],
            "embeddings": [
                "Normalize embeddings for consistent similarity computation",
                "Use appropriate embedding dimensions for your data size",
                "Implement proper batch processing for large datasets",
                "Cache embeddings when possible to improve performance"
            ],
            "search": [
                "Implement hybrid search combining multiple similarity methods",
                "Use appropriate indexing for your scale requirements",
                "Implement proper result ranking and relevance scoring",
                "Monitor search performance and user satisfaction"
            ],
            "security": [
                "Validate all inputs before processing",
                "Implement proper authentication and authorization",
                "Use secure communication protocols",
                "Monitor for security threats and anomalies"
            ],
            "performance": [
                "Profile code to identify bottlenecks",
                "Implement proper caching strategies",
                "Use asynchronous processing where appropriate",
                "Monitor system resources and scale accordingly"
            ]
        }
        
        for category, practices in default_practices.items():
            self.practices_db[category] = [
                {
                    "practice": practice,
                    "category": category,
                    "importance": "high",
                    "created_at": time.time()
                }
                for practice in practices
            ]
    
    def get_practices(self, component: str) -> List[str]:
        """Get best practices for a component."""
        component_lower = component.lower()
        
        # Direct category match
        if component_lower in self.practices_db:
            return [p["practice"] for p in self.practices_db[component_lower]]
        
        # Search across all categories
        matching_practices = []
        for category, practices in self.practices_db.items():
            if component_lower in category:
                matching_practices.extend([p["practice"] for p in practices])
        
        return matching_practices[:10]  # Return top 10
    
    def add_practice(self, category: str, practice: str, importance: str = "medium") -> None:
        """Add a new best practice."""
        if category not in self.practices_db:
            self.practices_db[category] = []
        
        self.practices_db[category].append({
            "practice": practice,
            "category": category,
            "importance": importance,
            "created_at": time.time()
        })


class QuantumRerankKnowledgeBase:
    """
    Comprehensive knowledge management system for QuantumRerank.
    
    Integrates documentation search, interactive examples, FAQs,
    and best practices into a unified knowledge base.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize knowledge base.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.search_engine = DocumentationSearchEngine(
            embedding_model=self.config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        self.code_examples = InteractiveCodeExamples()
        self.faq_system = FAQSystem()
        self.best_practices = BestPracticesDatabase()
        
        # Initialize documentation database
        self._initialize_documentation()
        
        self.logger = logger
        logger.info("Initialized QuantumRerankKnowledgeBase")
    
    def _initialize_documentation(self) -> None:
        """Initialize documentation database with core content."""
        # Core documentation items
        core_docs = [
            {
                "id": "quantum_overview",
                "title": "Quantum-Enhanced Information Retrieval Overview",
                "content": """QuantumRerank implements quantum-enhanced information retrieval using quantum fidelity computation and hybrid quantum-classical algorithms. The system combines classical embeddings with quantum similarity measures to improve search relevance and ranking accuracy.""",
                "doc_type": DocumentationType.TECHNICAL,
                "tags": ["quantum", "overview", "architecture"],
                "category": "fundamentals",
                "difficulty_level": "beginner"
            },
            {
                "id": "swap_test_algorithm",
                "title": "SWAP Test Algorithm Implementation",
                "content": """The SWAP test is a quantum algorithm used to compute the fidelity between two quantum states. In QuantumRerank, we use the SWAP test to measure similarity between quantum-encoded embeddings, providing a quantum advantage for certain similarity computations.""",
                "doc_type": DocumentationType.TECHNICAL,
                "tags": ["quantum", "swap-test", "algorithm", "fidelity"],
                "category": "algorithms",
                "difficulty_level": "intermediate"
            },
            {
                "id": "getting_started",
                "title": "Getting Started with QuantumRerank",
                "content": """This guide walks you through setting up QuantumRerank, creating your first quantum-enhanced search application, and understanding the core concepts. Start here if you're new to quantum-enhanced information retrieval.""",
                "doc_type": DocumentationType.TUTORIAL,
                "tags": ["tutorial", "getting-started", "setup"],
                "category": "tutorials",
                "difficulty_level": "beginner"
            },
            {
                "id": "api_reference",
                "title": "API Reference",
                "content": """Complete API reference for QuantumRerank including all classes, methods, and configuration options. Use this reference to understand the available functionality and integrate QuantumRerank into your applications.""",
                "doc_type": DocumentationType.API,
                "tags": ["api", "reference", "documentation"],
                "category": "reference",
                "difficulty_level": "intermediate"
            }
        ]
        
        for doc_data in core_docs:
            doc = DocumentationItem(**doc_data)
            self.search_engine.index_document(doc)
    
    def search_documentation(self, query: str, context: Optional[str] = None,
                           search_mode: SearchMode = SearchMode.HYBRID,
                           filters: Optional[Dict[str, Any]] = None) -> SearchResults:
        """
        Search comprehensive documentation with context awareness.
        
        Args:
            query: Search query
            context: Optional context for search
            search_mode: Search mode to use
            filters: Optional search filters
            
        Returns:
            SearchResults with ranked documentation
        """
        # Enhance query with context if provided
        enhanced_query = query
        if context:
            enhanced_query = f"{context} {query}"
        
        # Perform search
        results = self.search_engine.search(
            enhanced_query,
            mode=search_mode,
            filters=filters
        )
        
        return results
    
    def get_contextual_help(self, component: str, operation: str = "") -> ContextualHelp:
        """
        Provide contextual help for specific components and operations.
        
        Args:
            component: Component name
            operation: Optional operation name
            
        Returns:
            ContextualHelp with comprehensive information
        """
        # Get component description from documentation
        search_results = self.search_documentation(f"{component} {operation}", context="help")
        description = ""
        if search_results.results:
            description = search_results.results[0].item.content[:200] + "..."
        
        # Get usage examples
        usage_examples = self.code_examples.get_examples_for_topic(component)
        
        # Get common issues
        common_issues = self.faq_system.get_common_issues(component)
        
        # Get best practices
        best_practices = self.best_practices.get_practices(component)
        
        # Get related topics
        related_search = self.search_documentation(component, filters={"limit": 5})
        related_topics = [result.item.title for result in related_search.results[1:]]  # Skip first (self)
        
        # Generate quick actions
        quick_actions = self._generate_quick_actions(component, operation)
        
        return ContextualHelp(
            component=component,
            operation=operation,
            description=description,
            usage_examples=usage_examples,
            common_issues=common_issues,
            best_practices=best_practices,
            related_topics=related_topics,
            quick_actions=quick_actions
        )
    
    def _generate_quick_actions(self, component: str, operation: str) -> List[Dict[str, str]]:
        """Generate quick actions for component and operation."""
        quick_actions = []
        
        # Common quick actions based on component
        if "quantum" in component.lower():
            quick_actions.append({
                "title": "View Quantum Circuit",
                "action": f"show_quantum_circuit({component})"
            })
            quick_actions.append({
                "title": "Run Quantum Simulation",
                "action": f"simulate_quantum({component})"
            })
        
        if "embedding" in component.lower():
            quick_actions.append({
                "title": "Generate Embeddings",
                "action": f"generate_embeddings({component})"
            })
            quick_actions.append({
                "title": "Compute Similarity",
                "action": f"compute_similarity({component})"
            })
        
        if "search" in component.lower():
            quick_actions.append({
                "title": "Perform Search",
                "action": f"search_documents({component})"
            })
            quick_actions.append({
                "title": "Rerank Results",
                "action": f"rerank_results({component})"
            })
        
        # Generic actions
        quick_actions.extend([
            {
                "title": "View Examples",
                "action": f"show_examples({component})"
            },
            {
                "title": "Open Documentation",
                "action": f"open_docs({component})"
            }
        ])
        
        return quick_actions[:5]  # Limit to 5 actions
    
    def add_documentation(self, doc: DocumentationItem) -> None:
        """Add new documentation item to knowledge base."""
        self.search_engine.index_document(doc)
    
    def add_code_example(self, example_id: str, title: str, code: str,
                        description: str, **kwargs) -> None:
        """Add new interactive code example."""
        self.code_examples.add_example(example_id, title, code, description, **kwargs)
    
    def add_faq(self, faq_id: str, question: str, answer: str, **kwargs) -> None:
        """Add new FAQ item."""
        self.faq_system.add_faq(faq_id, question, answer, **kwargs)
    
    def get_knowledge_base_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge base statistics."""
        return {
            "documentation": {
                "total_items": len(self.search_engine.document_index),
                "search_analytics": self.search_engine.get_search_analytics()
            },
            "code_examples": {
                "total_examples": len(self.code_examples.examples_db),
                "total_executions": sum(ex["execution_count"] for ex in self.code_examples.examples_db.values())
            },
            "faqs": {
                "total_faqs": len(self.faq_system.faq_database),
                "total_views": sum(faq["view_count"] for faq in self.faq_system.faq_database.values())
            },
            "best_practices": {
                "total_categories": len(self.best_practices.practices_db),
                "total_practices": sum(len(practices) for practices in self.best_practices.practices_db.values())
            }
        }