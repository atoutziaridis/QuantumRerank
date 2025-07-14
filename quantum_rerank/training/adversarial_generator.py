"""
Adversarial Example Generator for Quantum Similarity Training.

Generates harder training examples to improve quantum model robustness
without adding unnecessary complexity to the core system.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import random
from dataclasses import dataclass
import logging

from ..core.embeddings import EmbeddingProcessor

logger = logging.getLogger(__name__)

@dataclass
class AdversarialConfig:
    """Configuration for adversarial example generation."""
    difficulty_levels: List[str] = None  # ["easy", "medium", "hard"]
    perturbation_strength: float = 0.1  # How much to perturb embeddings
    num_hard_negatives: int = 5  # Number of hard negatives per positive
    similarity_threshold: float = 0.7  # Threshold for hard negative selection
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ["easy", "medium", "hard"]

class AdversarialGenerator:
    """Generate adversarial training examples for quantum similarity learning."""
    
    def __init__(self, config: AdversarialConfig = None):
        self.config = config or AdversarialConfig()
        self.embedding_processor = EmbeddingProcessor()
        
        # Predefined challenging query-document pairs
        self.challenge_templates = {
            "easy": [
                ("cat", "dog"),  # Different animals
                ("car", "bicycle"),  # Different vehicles
                ("apple", "orange"),  # Different fruits
            ],
            "medium": [
                ("machine learning", "artificial intelligence"),  # Related concepts
                ("deep learning", "neural networks"),  # Overlapping terms
                ("quantum computing", "classical computing"),  # Contrasting approaches
            ],
            "hard": [
                ("The quick brown fox", "A fast auburn canine"),  # Paraphrases
                ("Neural networks learn patterns", "Artificial networks recognize trends"),  # Similar meaning, different words
                ("Quantum supremacy achieved", "Classical computers outperformed"),  # Opposing meanings
            ]
        }
        
        logger.info(f"Adversarial generator initialized with {len(self.challenge_templates)} template categories")
    
    def generate_triplets(self, base_documents: List[str], difficulty: str = "medium", 
                         num_triplets: int = 100) -> List[Tuple[str, str, str]]:
        """
        Generate adversarial triplets (query, positive, negative).
        
        Args:
            base_documents: Base document collection
            difficulty: Difficulty level ("easy", "medium", "hard")
            num_triplets: Number of triplets to generate
            
        Returns:
            List of (query, positive, negative) triplets
        """
        if difficulty not in self.config.difficulty_levels:
            raise ValueError(f"Invalid difficulty: {difficulty}")
        
        triplets = []
        
        # Use templates for structured adversarial examples
        templates = self.challenge_templates.get(difficulty, [])
        
        for i in range(num_triplets):
            if i < len(templates) * 10:  # Use templates for first portion
                template_idx = i % len(templates)
                query, positive = templates[template_idx]
                
                # Generate hard negative
                negative = self._generate_hard_negative(query, positive, base_documents)
                
            else:  # Generate from base documents
                query, positive, negative = self._generate_from_documents(base_documents, difficulty)
            
            triplets.append((query, positive, negative))
        
        logger.info(f"Generated {len(triplets)} {difficulty} adversarial triplets")
        return triplets
    
    def _generate_hard_negative(self, query: str, positive: str, 
                              base_documents: List[str]) -> str:
        """Generate a hard negative example that's similar but not correct."""
        
        # Get embeddings for query and positive
        embeddings = self.embedding_processor.encode_texts([query, positive])
        query_emb, pos_emb = embeddings[0], embeddings[1]
        
        # Find documents that are similar to query but different from positive
        best_negative = None
        best_score = -1
        
        # Sample from base documents
        sample_docs = random.sample(base_documents, min(50, len(base_documents)))
        
        for doc in sample_docs:
            doc_emb = self.embedding_processor.encode_texts([doc])[0]
            
            # Score: similar to query, dissimilar to positive
            query_sim = self.embedding_processor.compute_classical_similarity(query_emb, doc_emb)
            pos_sim = self.embedding_processor.compute_classical_similarity(pos_emb, doc_emb)
            
            # Hard negative score: high similarity to query, low to positive
            score = query_sim - pos_sim
            
            if score > best_score:
                best_score = score
                best_negative = doc
        
        return best_negative or random.choice(base_documents)
    
    def _generate_from_documents(self, documents: List[str], difficulty: str) -> Tuple[str, str, str]:
        """Generate triplet from document collection based on difficulty."""
        
        # Sample documents
        sample_size = min(20, len(documents))
        sample_docs = random.sample(documents, sample_size)
        
        # Choose query and positive
        query = random.choice(sample_docs)
        
        if difficulty == "easy":
            # Easy: clearly different documents
            remaining = [d for d in sample_docs if d != query]
            positive = self._find_most_similar(query, remaining[:5])
            negative = self._find_least_similar(query, remaining[-5:])
            
        elif difficulty == "medium":
            # Medium: moderately similar documents
            remaining = [d for d in sample_docs if d != query]
            positive = self._find_most_similar(query, remaining)
            # Find negative that's somewhat similar but less than positive
            negative = self._find_medium_similarity(query, positive, remaining)
            
        else:  # hard
            # Hard: very similar documents requiring fine-grained distinction
            remaining = [d for d in sample_docs if d != query]
            positive = self._find_most_similar(query, remaining)
            # Find negative very similar to positive
            negative = self._find_confusing_negative(query, positive, remaining)
        
        return query, positive, negative
    
    def _find_most_similar(self, query: str, candidates: List[str]) -> str:
        """Find most similar document to query."""
        if not candidates:
            return query
        
        query_emb = self.embedding_processor.encode_texts([query])[0]
        best_sim = -1
        best_doc = candidates[0]
        
        for doc in candidates:
            doc_emb = self.embedding_processor.encode_texts([doc])[0]
            sim = self.embedding_processor.compute_classical_similarity(query_emb, doc_emb)
            
            if sim > best_sim:
                best_sim = sim
                best_doc = doc
        
        return best_doc
    
    def _find_least_similar(self, query: str, candidates: List[str]) -> str:
        """Find least similar document to query."""
        if not candidates:
            return query
        
        query_emb = self.embedding_processor.encode_texts([query])[0]
        best_sim = 2.0  # Start high
        best_doc = candidates[0]
        
        for doc in candidates:
            doc_emb = self.embedding_processor.encode_texts([doc])[0]
            sim = self.embedding_processor.compute_classical_similarity(query_emb, doc_emb)
            
            if sim < best_sim:
                best_sim = sim
                best_doc = doc
        
        return best_doc
    
    def _find_medium_similarity(self, query: str, positive: str, candidates: List[str]) -> str:
        """Find document with medium similarity - harder than easy but not confusing."""
        if not candidates:
            return positive
        
        query_emb = self.embedding_processor.encode_texts([query])[0]
        pos_emb = self.embedding_processor.encode_texts([positive])[0]
        pos_sim = self.embedding_processor.compute_classical_similarity(query_emb, pos_emb)
        
        target_sim = pos_sim * 0.7  # Target 70% of positive similarity
        best_doc = candidates[0]
        best_diff = float('inf')
        
        for doc in candidates:
            if doc == positive:
                continue
                
            doc_emb = self.embedding_processor.encode_texts([doc])[0]
            sim = self.embedding_processor.compute_classical_similarity(query_emb, doc_emb)
            
            diff = abs(sim - target_sim)
            if diff < best_diff:
                best_diff = diff
                best_doc = doc
        
        return best_doc
    
    def _find_confusing_negative(self, query: str, positive: str, candidates: List[str]) -> str:
        """Find negative that's very similar to positive - most confusing case."""
        if not candidates:
            return positive
        
        pos_emb = self.embedding_processor.encode_texts([positive])[0]
        best_sim = -1
        best_doc = candidates[0]
        
        for doc in candidates:
            if doc == positive:
                continue
                
            doc_emb = self.embedding_processor.encode_texts([doc])[0]
            sim = self.embedding_processor.compute_classical_similarity(pos_emb, doc_emb)
            
            if sim > best_sim:
                best_sim = sim
                best_doc = doc
        
        return best_doc
    
    def curriculum_learning_schedule(self, epoch: int, total_epochs: int) -> str:
        """Determine difficulty level based on training progress."""
        progress = epoch / total_epochs
        
        if progress < 0.3:
            return "easy"
        elif progress < 0.7:
            return "medium"
        else:
            return "hard"
    
    def generate_curriculum_batch(self, base_documents: List[str], epoch: int, 
                                total_epochs: int, batch_size: int = 32) -> List[Tuple[str, str, str]]:
        """Generate a batch of triplets following curriculum learning."""
        
        difficulty = self.curriculum_learning_schedule(epoch, total_epochs)
        triplets = self.generate_triplets(base_documents, difficulty, batch_size)
        
        logger.debug(f"Epoch {epoch}/{total_epochs}: Generated {len(triplets)} {difficulty} triplets")
        return triplets