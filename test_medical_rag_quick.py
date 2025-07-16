"""
Quick medical RAG test comparing classical vs quantum-enhanced retrieval.
Focuses on demonstrating quantum advantages with noisy medical data.
"""

import numpy as np
import time
from typing import List, Dict
import json

from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata
from quantum_rerank.core.embeddings import EmbeddingProcessor


def create_noisy_medical_docs() -> List[Document]:
    """Create a small set of medical documents with realistic noise."""
    
    # Clean medical content
    medical_contents = {
        "cardio_1": "Patient presents with hypertension and elevated blood pressure readings. Cardiac examination reveals irregular rhythm. ECG shows atrial fibrillation. Recommend beta blockers and anticoagulation therapy.",
        
        "cardio_2": "Coronary artery disease confirmed via angiography. Patient experiencing angina pectoris. Myocardial perfusion scan indicates reduced blood flow. Considering coronary bypass surgery.",
        
        "diabetes_1": "Type 2 diabetes mellitus with poor glycemic control. HbA1c level at 9.2%. Patient reports neuropathy symptoms. Initiating insulin therapy and continuous glucose monitoring.",
        
        "diabetes_2": "Diabetic ketoacidosis presentation. Blood glucose critically elevated at 450 mg/dL. Metabolic acidosis present. Emergency insulin infusion protocol initiated.",
        
        "respiratory_1": "COPD exacerbation with severe dyspnea. Pulmonary function tests show reduced FEV1. Oxygen saturation at 88%. Started on bronchodilators and systemic corticosteroids.",
        
        "respiratory_2": "Pneumonia confirmed on chest X-ray. Productive cough with purulent sputum. Fever and elevated white blood cell count. Initiated antibiotic therapy.",
        
        "neuro_1": "Acute ischemic stroke presentation. CT scan shows cerebral infarction. NIH stroke scale score of 12. Administering thrombolytic therapy within window.",
        
        "neuro_2": "Epilepsy with frequent seizures. EEG shows abnormal spike-wave discharges. Poor response to current anticonvulsants. Adjusting medication regimen.",
        
        "oncology_1": "Breast cancer stage IIIA confirmed. Tumor markers elevated. PET scan shows lymph node involvement. Planning neoadjuvant chemotherapy followed by surgery.",
        
        "oncology_2": "Lung cancer with brain metastases. Performance status declining. Considering palliative radiation therapy and immunotherapy options."
    }
    
    # Add realistic noise patterns
    def add_noise(text: str) -> str:
        # OCR-like errors
        replacements = {
            'a': ['e', 'o'], 'e': ['a', 'c'], 'o': ['0', 'e'],
            'l': ['1', 'i'], 's': ['5', 'z'], 'g': ['9', 'q']
        }
        
        noisy = text
        for char, subs in replacements.items():
            if np.random.random() < 0.1:  # 10% chance per character type
                noisy = noisy.replace(char, np.random.choice(subs), 1)
        
        # Medical abbreviation variations
        if "blood pressure" in noisy and np.random.random() < 0.5:
            noisy = noisy.replace("blood pressure", "BP")
        if "electrocardiogram" in noisy and np.random.random() < 0.5:
            noisy = noisy.replace("electrocardiogram", "ECG")
            
        return noisy
    
    # Create documents
    documents = []
    for doc_id, (key, content) in enumerate(medical_contents.items()):
        topic = key.split('_')[0]
        noisy_content = add_noise(content)
        
        doc = Document(
            doc_id=f"med_{doc_id}",
            content=noisy_content,
            metadata=DocumentMetadata(custom_fields={
                "topic": topic,
                "original_key": key
            })
        )
        documents.append(doc)
    
    return documents


def run_quick_comparison():
    """Run a quick comparison test."""
    print("\n" + "="*60)
    print("Medical RAG Test - Classical vs Quantum")
    print("="*60 + "\n")
    
    # Initialize system
    from quantum_rerank.retrieval.two_stage_retriever import RetrieverConfig
    config = RetrieverConfig(
        initial_k=5,  # Smaller for faster testing
        final_k=3,
        reranking_method="hybrid"
    )
    retriever = TwoStageRetriever(config=config)
    
    # Create and add documents
    print("Creating noisy medical documents...")
    docs = create_noisy_medical_docs()
    print(f"Created {len(docs)} documents with OCR-like noise\n")
    
    retriever.add_documents(docs)
    
    # Test queries
    queries = [
        {
            "text": "patient with high blood pressure and heart problems",
            "expected": "cardio"
        },
        {
            "text": "diabetes with poor sugar control needing insulin",
            "expected": "diabetes"
        },
        {
            "text": "breathing difficulties and lung disease",
            "expected": "respiratory"
        }
    ]
    
    results_summary = {
        "classical": {"correct": 0, "total_time": 0},
        "quantum": {"correct": 0, "total_time": 0}
    }
    
    print("Running retrieval tests...")
    print("-" * 60)
    
    embedding_processor = retriever.embedding_processor
    
    for i, query_data in enumerate(queries):
        query = query_data["text"]
        expected_topic = query_data["expected"]
        
        print(f"\nQuery {i+1}: {query}")
        print(f"Expected topic: {expected_topic}")
        
        # Classical retrieval (FAISS only)
        query_embedding = embedding_processor.encode_single_text(query)
        
        start = time.time()
        faiss_results = retriever.faiss_store.search(query_embedding, k=3)
        classical_time = (time.time() - start) * 1000
        
        # Check classical results
        classical_topics = []
        for result in faiss_results:
            doc = retriever.document_store.get_document(result.doc_id)
            if doc:
                topic = doc.metadata.custom_fields.get("topic", "unknown")
                classical_topics.append(topic)
        
        classical_correct = expected_topic in classical_topics
        if classical_correct:
            results_summary["classical"]["correct"] += 1
        results_summary["classical"]["total_time"] += classical_time
        
        # Quantum-enhanced retrieval
        start = time.time()
        quantum_results = retriever.retrieve(query, k=3)
        quantum_time = (time.time() - start) * 1000
        
        # Check quantum results
        quantum_topics = []
        for result in quantum_results:
            doc = retriever.document_store.get_document(result.doc_id)
            if doc:
                topic = doc.metadata.custom_fields.get("topic", "unknown")
                quantum_topics.append(topic)
        
        quantum_correct = expected_topic in quantum_topics
        if quantum_correct:
            results_summary["quantum"]["correct"] += 1
        results_summary["quantum"]["total_time"] += quantum_time
        
        # Print results
        print(f"  Classical: {classical_topics} - {'✓' if classical_correct else '✗'} ({classical_time:.1f}ms)")
        print(f"  Quantum:   {quantum_topics} - {'✓' if quantum_correct else '✗'} ({quantum_time:.1f}ms)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    classical_accuracy = results_summary["classical"]["correct"] / len(queries) * 100
    quantum_accuracy = results_summary["quantum"]["correct"] / len(queries) * 100
    
    print(f"\nClassical FAISS:")
    print(f"  Accuracy: {classical_accuracy:.0f}% ({results_summary['classical']['correct']}/{len(queries)})")
    print(f"  Avg Time: {results_summary['classical']['total_time']/len(queries):.1f}ms")
    
    print(f"\nQuantum-Enhanced:")
    print(f"  Accuracy: {quantum_accuracy:.0f}% ({results_summary['quantum']['correct']}/{len(queries)})")
    print(f"  Avg Time: {results_summary['quantum']['total_time']/len(queries):.1f}ms")
    
    print(f"\nImprovement: {quantum_accuracy - classical_accuracy:+.0f}% accuracy")
    
    # Test on noisy similarity
    print("\n" + "="*60)
    print("NOISE ROBUSTNESS TEST")
    print("="*60)
    
    # Get two similar cardio documents
    cardio_docs = [d for d in docs if "cardio" in d.metadata.custom_fields.get("original_key", "")]
    if len(cardio_docs) >= 2:
        doc1, doc2 = cardio_docs[0], cardio_docs[1]
        
        # Get embeddings
        emb1 = embedding_processor.encode_single_text(doc1.content)
        emb2 = embedding_processor.encode_single_text(doc2.content)
        
        # Classical cosine similarity
        classical_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Quantum similarity (using retriever's quantum engine)
        from quantum_rerank.core.quantum_similarity_engine import QuantumSimilarityEngine, SimilarityMethod
        quantum_engine = QuantumSimilarityEngine()
        quantum_sim = quantum_engine.compute_similarity(emb1, emb2, method=SimilarityMethod.QUANTUM)
        
        print(f"\nSimilarity between two cardiac documents (with noise):")
        print(f"  Classical Cosine: {classical_sim:.3f}")
        print(f"  Quantum Fidelity: {quantum_sim:.3f}")
        print(f"  Quantum captures {(quantum_sim/classical_sim - 1)*100:+.0f}% more similarity")


if __name__ == "__main__":
    run_quick_comparison()