#!/usr/bin/env python3
"""
Debug Retrieval Pipeline
Investigate why FAISS is returning no results.
"""

import logging
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from quantum_rerank.core.embeddings import EmbeddingProcessor, EmbeddingConfig
from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever, RetrieverConfig
from quantum_rerank.retrieval.faiss_store import QuantumFAISSStore, FAISSConfig
from quantum_rerank.retrieval.document_store import DocumentStore, Document

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_retrieval_pipeline():
    """Debug the entire retrieval pipeline step by step."""
    logger.info("=== DEBUGGING RETRIEVAL PIPELINE ===")
    
    # Step 1: Test embedding generation
    logger.info("\n1. Testing embedding generation...")
    embedding_processor = EmbeddingProcessor(EmbeddingConfig())
    
    test_texts = [
        "Patient with chest pain and shortness of breath",
        "Diabetes management guidelines",
        "Hypertension treatment protocols",
        "Emergency cardiac procedures",
        "Chest pain differential diagnosis"
    ]
    
    embeddings = embedding_processor.encode_texts(test_texts)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"Sample embedding values: {embeddings[0][:5]}")
    
    # Step 2: Test DocumentStore
    logger.info("\n2. Testing DocumentStore...")
    doc_store = DocumentStore(embedding_processor)
    
    # Add texts to document store
    doc_ids = doc_store.add_texts(test_texts)
    logger.info(f"Added {len(doc_ids)} documents to store")
    logger.info(f"Document IDs: {doc_ids}")
    
    # Check documents have embeddings
    for doc_id in doc_ids:
        doc = doc_store.get_document(doc_id)
        if doc and doc.embedding:
            logger.info(f"Document {doc_id}: has embedding of length {len(doc.embedding)}")
        else:
            logger.error(f"Document {doc_id}: missing embedding!")
    
    # Step 3: Test FAISS Store directly
    logger.info("\n3. Testing FAISS Store directly...")
    faiss_config = FAISSConfig(dimension=embeddings.shape[1])
    faiss_store = QuantumFAISSStore(faiss_config)
    
    # Add embeddings to FAISS
    metadatas = [{"text": text, "relevance": 0.8} for text in test_texts]
    added = faiss_store.add_documents(embeddings, doc_ids, metadatas)
    logger.info(f"Added {added} documents to FAISS")
    logger.info(f"FAISS stats: {faiss_store.get_stats()}")
    
    # Test search
    query_text = "chest pain diagnosis"
    query_embedding = embedding_processor.encode_texts([query_text])[0]
    logger.info(f"Query embedding shape: {query_embedding.shape}")
    
    search_results = faiss_store.search(query_embedding, k=3)
    logger.info(f"FAISS search returned {len(search_results)} results")
    
    for i, result in enumerate(search_results):
        logger.info(f"Result {i+1}: doc_id={result.doc_id}, score={result.score:.4f}")
    
    # Step 4: Test TwoStageRetriever
    logger.info("\n4. Testing TwoStageRetriever...")
    config = RetrieverConfig(initial_k=10, final_k=3, reranking_method="classical")
    retriever = TwoStageRetriever(config=config, embedding_processor=embedding_processor)
    
    # Clear and add documents
    retriever.clear()
    doc_ids = retriever.add_texts(test_texts, metadatas)
    logger.info(f"Added {len(doc_ids)} documents to retriever")
    
    # Test retrieval
    results = retriever.retrieve(query_text, k=3)
    logger.info(f"Retriever returned {len(results)} results")
    
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: doc_id={result.doc_id}, score={result.score:.4f}, content={result.content[:50]}...")
    
    # Step 5: Test with real medical scenario
    logger.info("\n5. Testing with medical scenario...")
    medical_texts = [
        "Pulmonary embolism diagnosis in patients with chest pain and dyspnea. Clinical presentation includes acute onset chest pain, shortness of breath, and fatigue. Risk factors include recent travel, immobilization, and hypercoagulable states.",
        "Acute coronary syndrome evaluation in chest pain patients. Symptoms include crushing chest pain, dyspnea, diaphoresis, and fatigue. ECG findings, troponin levels, and risk stratification are essential.",
        "Anxiety disorders presenting with somatic symptoms. Panic attacks can mimic cardiac conditions with chest pain, palpitations, and shortness of breath.",
        "Diabetes management and glycemic control. HbA1c targets, medication selection, and lifestyle interventions for type 2 diabetes mellitus.",
        "Hypertension guidelines and treatment protocols. Blood pressure targets, antihypertensive medications, and cardiovascular risk reduction strategies."
    ]
    
    relevance_scores = [0.95, 0.90, 0.60, 0.10, 0.15]
    medical_metadatas = [{"relevance": score} for score in relevance_scores]
    
    # Create new retriever for medical test
    retriever.clear()
    medical_doc_ids = retriever.add_texts(medical_texts, medical_metadatas)
    
    medical_query = "Patient presenting with chest pain, shortness of breath, and fatigue. Recent travel history. What are the most likely diagnoses?"
    medical_results = retriever.retrieve(medical_query, k=3)
    
    logger.info(f"Medical scenario returned {len(medical_results)} results")
    for i, result in enumerate(medical_results):
        logger.info(f"Medical result {i+1}: score={result.score:.4f}, content={result.content[:80]}...")
    
    # Calculate metrics if we got results
    if medical_results:
        retrieved_relevances = []
        for result in medical_results:
            # Find the relevance score from metadata
            if hasattr(result, 'metadata') and 'relevance' in result.metadata:
                retrieved_relevances.append(result.metadata['relevance'])
            else:
                # Match by content
                for i, text in enumerate(medical_texts):
                    if text == result.content:
                        retrieved_relevances.append(relevance_scores[i])
                        break
                else:
                    retrieved_relevances.append(0.0)
        
        logger.info(f"Retrieved relevances: {retrieved_relevances}")
        
        # Calculate NDCG@3
        ideal_relevances = sorted(relevance_scores, reverse=True)[:3]
        logger.info(f"Ideal relevances: {ideal_relevances}")
        
        def dcg(relevances):
            return sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances)])
        
        dcg_score = dcg(retrieved_relevances)
        idcg_score = dcg(ideal_relevances)
        ndcg = dcg_score / idcg_score if idcg_score > 0 else 0
        
        logger.info(f"NDCG@3: {ndcg:.4f}")
        
        if ndcg > 0:
            logger.info("✅ SUCCESS: Retrieval pipeline is working!")
        else:
            logger.warning("⚠️  WARNING: NDCG is 0, ranking may be poor")
    else:
        logger.error("❌ FAILURE: No results returned from medical scenario")

if __name__ == "__main__":
    debug_retrieval_pipeline()