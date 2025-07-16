"""
Real-world medical RAG testing with noise injection.
Compares classical RAG vs classical + quantum reranker on noisy medical data.
"""

import numpy as np
import random
import string
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import json

from quantum_rerank.retrieval.two_stage_retriever import TwoStageRetriever
from quantum_rerank.retrieval.document_store import Document, DocumentMetadata
from quantum_rerank.core.quantum_kernel_engine import QuantumKernelEngine
from quantum_rerank.core.embeddings import EmbeddingProcessor


@dataclass
class TestResult:
    method: str
    query: str
    top_k_accuracy: float
    mrr: float  # Mean Reciprocal Rank
    latency_ms: float
    relevance_scores: List[float]


class MedicalDocumentGenerator:
    """Generate realistic medical documents with various topics."""
    
    def __init__(self):
        self.medical_topics = {
            "cardiovascular": {
                "terms": ["hypertension", "blood pressure", "cardiac", "heart", "arrhythmia", 
                         "coronary", "myocardial", "atherosclerosis", "angina", "ECG"],
                "conditions": ["heart failure", "atrial fibrillation", "myocardial infarction",
                             "coronary artery disease", "hypertensive crisis"],
                "treatments": ["beta blockers", "ACE inhibitors", "statins", "anticoagulants",
                             "cardiac catheterization", "angioplasty", "bypass surgery"]
            },
            "diabetes": {
                "terms": ["glucose", "insulin", "glycemic", "HbA1c", "hyperglycemia",
                         "pancreas", "metabolic", "ketoacidosis", "neuropathy", "retinopathy"],
                "conditions": ["Type 1 diabetes", "Type 2 diabetes", "gestational diabetes",
                             "diabetic ketoacidosis", "hypoglycemia"],
                "treatments": ["metformin", "insulin therapy", "continuous glucose monitoring",
                             "dietary management", "exercise therapy", "GLP-1 agonists"]
            },
            "respiratory": {
                "terms": ["pulmonary", "lung", "breathing", "oxygen", "bronchial",
                         "alveolar", "respiratory", "ventilation", "spirometry", "asthma"],
                "conditions": ["COPD", "pneumonia", "asthma", "pulmonary embolism",
                             "acute respiratory distress syndrome", "bronchitis"],
                "treatments": ["bronchodilators", "corticosteroids", "oxygen therapy",
                             "mechanical ventilation", "pulmonary rehabilitation", "antibiotics"]
            },
            "neurological": {
                "terms": ["neural", "brain", "cognitive", "seizure", "neurological",
                         "cerebral", "motor", "sensory", "reflex", "EEG"],
                "conditions": ["stroke", "epilepsy", "Parkinson's disease", "multiple sclerosis",
                             "migraine", "Alzheimer's disease", "peripheral neuropathy"],
                "treatments": ["anticonvulsants", "dopamine agonists", "thrombolytics",
                             "physical therapy", "occupational therapy", "deep brain stimulation"]
            },
            "oncology": {
                "terms": ["tumor", "cancer", "malignant", "metastasis", "oncology",
                         "carcinoma", "lymphoma", "chemotherapy", "radiation", "biopsy"],
                "conditions": ["breast cancer", "lung cancer", "colorectal cancer",
                             "leukemia", "lymphoma", "melanoma", "prostate cancer"],
                "treatments": ["chemotherapy", "radiation therapy", "immunotherapy",
                             "targeted therapy", "surgery", "hormone therapy", "stem cell transplant"]
            }
        }
    
    def generate_medical_document(self, topic: str, doc_id: int) -> str:
        """Generate a realistic medical document for a given topic."""
        topic_data = self.medical_topics[topic]
        
        # Create structured medical document
        sections = []
        
        # Patient presentation
        condition = random.choice(topic_data["conditions"])
        age = random.randint(25, 85)
        gender = random.choice(["male", "female"])
        
        sections.append(f"Patient ID: {doc_id:05d}")
        sections.append(f"Demographics: {age}-year-old {gender}")
        sections.append(f"Chief Complaint: Patient presents with symptoms consistent with {condition}.")
        
        # Clinical findings
        findings = []
        for _ in range(3):
            term = random.choice(topic_data["terms"])
            value = random.choice(["elevated", "decreased", "within normal limits", "abnormal"])
            findings.append(f"{term} levels {value}")
        
        sections.append(f"Clinical Findings: {', '.join(findings)}.")
        
        # Diagnosis
        sections.append(f"Primary Diagnosis: {condition}")
        if random.random() > 0.5:
            secondary = random.choice([c for c in topic_data["conditions"] if c != condition])
            sections.append(f"Secondary Diagnosis: {secondary}")
        
        # Treatment plan
        treatments = random.sample(topic_data["treatments"], min(3, len(topic_data["treatments"])))
        sections.append(f"Treatment Plan: Initiated {', '.join(treatments)}.")
        
        # Follow-up
        sections.append(f"Follow-up: Patient scheduled for reassessment in {random.choice([1, 2, 4, 8])} weeks.")
        
        # Add some medical jargon
        jargon = [
            "Vital signs stable", "No acute distress", "Alert and oriented",
            "Regular rate and rhythm", "Clear to auscultation bilaterally",
            "Soft, non-tender, non-distended", "No peripheral edema"
        ]
        sections.append(f"Additional Notes: {random.choice(jargon)}.")
        
        return " ".join(sections)
    
    def generate_dataset(self, docs_per_topic: int = 20) -> List[Document]:
        """Generate a dataset of medical documents."""
        documents = []
        doc_id = 1000
        
        for topic in self.medical_topics:
            for i in range(docs_per_topic):
                content = self.generate_medical_document(topic, doc_id + i)
                doc = Document(
                    doc_id=f"med_{doc_id + i}",
                    content=content,
                    metadata=DocumentMetadata(custom_fields={"topic": topic, "doc_type": "clinical_note"})
                )
                documents.append(doc)
            doc_id += docs_per_topic
        
        return documents


class NoiseInjector:
    """Inject realistic noise into medical documents."""
    
    @staticmethod
    def add_ocr_errors(text: str, error_rate: float = 0.02) -> str:
        """Simulate OCR errors."""
        ocr_substitutions = {
            'a': ['e', 'o', 's'],
            'e': ['a', 'c', 'o'],
            'i': ['l', '1', 'j'],
            'o': ['0', 'e', 'a'],
            'l': ['1', 'i', '|'],
            's': ['5', 'z', '$'],
            'g': ['9', 'q', 'y'],
            'b': ['6', 'h', 'd'],
            'm': ['n', 'rn', 'nn'],
            'n': ['m', 'h', 'r'],
            '0': ['O', 'o', 'Q'],
            '1': ['l', 'I', '|'],
            '5': ['S', 's', '$']
        }
        
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < error_rate:
                char = chars[i].lower()
                if char in ocr_substitutions:
                    chars[i] = random.choice(ocr_substitutions[char])
        
        return ''.join(chars)
    
    @staticmethod
    def add_typos(text: str, error_rate: float = 0.015) -> str:
        """Add realistic typing errors."""
        chars = list(text)
        for i in range(len(chars) - 1):
            if random.random() < error_rate:
                # Transpose adjacent characters
                if chars[i].isalpha() and chars[i+1].isalpha():
                    chars[i], chars[i+1] = chars[i+1], chars[i]
        
        # Missing characters
        new_chars = []
        for char in chars:
            if random.random() > error_rate or not char.isalpha():
                new_chars.append(char)
        
        return ''.join(new_chars)
    
    @staticmethod
    def add_abbreviation_variations(text: str) -> str:
        """Add medical abbreviation variations."""
        abbreviations = {
            "blood pressure": ["BP", "B/P", "b.p."],
            "heart rate": ["HR", "h.r.", "pulse"],
            "temperature": ["temp", "T", "temp."],
            "respiratory rate": ["RR", "resp rate", "resp"],
            "oxygen saturation": ["O2 sat", "SpO2", "sats"],
            "electrocardiogram": ["ECG", "EKG", "electro"],
            "magnetic resonance imaging": ["MRI", "MR imaging", "magnetic res"],
            "computed tomography": ["CT", "CAT scan", "CT scan"]
        }
        
        result = text
        for full_term, abbrevs in abbreviations.items():
            if full_term in result and random.random() > 0.5:
                result = result.replace(full_term, random.choice(abbrevs), 1)
        
        return result
    
    @staticmethod
    def add_mixed_content(text: str, mix_rate: float = 0.1) -> str:
        """Mix content from different parts of the document."""
        sentences = text.split('. ')
        if len(sentences) > 3 and random.random() < mix_rate:
            # Randomly shuffle a portion of sentences
            start = random.randint(0, len(sentences) - 3)
            subset = sentences[start:start+3]
            random.shuffle(subset)
            sentences[start:start+3] = subset
        
        return '. '.join(sentences)
    
    @staticmethod
    def inject_all_noise(text: str, noise_level: str = "medium") -> str:
        """Apply all noise types based on noise level."""
        noise_configs = {
            "low": {"ocr": 0.005, "typo": 0.005, "mix": 0.05},
            "medium": {"ocr": 0.015, "typo": 0.01, "mix": 0.1},
            "high": {"ocr": 0.03, "typo": 0.02, "mix": 0.2}
        }
        
        config = noise_configs.get(noise_level, noise_configs["medium"])
        
        # Apply noise in sequence
        noisy_text = text
        noisy_text = NoiseInjector.add_ocr_errors(noisy_text, config["ocr"])
        noisy_text = NoiseInjector.add_typos(noisy_text, config["typo"])
        noisy_text = NoiseInjector.add_abbreviation_variations(noisy_text)
        noisy_text = NoiseInjector.add_mixed_content(noisy_text, config["mix"])
        
        return noisy_text


class MedicalRAGTester:
    """Test RAG systems on medical documents with noise."""
    
    def __init__(self):
        self.doc_generator = MedicalDocumentGenerator()
        self.noise_injector = NoiseInjector()
        self.retriever = TwoStageRetriever()
        self.quantum_engine = QuantumKernelEngine()
        self.embedding_processor = EmbeddingProcessor()
    
    def prepare_test_data(self, docs_per_topic: int = 20, noise_level: str = "medium") -> Tuple[List[Document], List[Dict]]:
        """Prepare test documents and queries."""
        # Generate clean documents
        clean_docs = self.doc_generator.generate_dataset(docs_per_topic)
        
        # Add noise to documents
        noisy_docs = []
        for doc in clean_docs:
            noisy_content = self.noise_injector.inject_all_noise(doc.content, noise_level)
            noisy_doc = Document(
                doc_id=doc.doc_id,
                content=noisy_content,
                metadata=doc.metadata
            )
            noisy_docs.append(noisy_doc)
        
        # Generate test queries
        queries = [
            {
                "query": "patient with elevated blood pressure and cardiac symptoms",
                "relevant_topic": "cardiovascular",
                "expected_terms": ["hypertension", "cardiac", "heart"]
            },
            {
                "query": "diabetic patient with poor glycemic control requiring insulin",
                "relevant_topic": "diabetes",
                "expected_terms": ["glucose", "insulin", "HbA1c"]
            },
            {
                "query": "respiratory distress with abnormal lung findings",
                "relevant_topic": "respiratory",
                "expected_terms": ["pulmonary", "lung", "breathing"]
            },
            {
                "query": "neurological symptoms including seizures and cognitive decline",
                "relevant_topic": "neurological",
                "expected_terms": ["neural", "seizure", "cognitive"]
            },
            {
                "query": "cancer treatment options including chemotherapy and radiation",
                "relevant_topic": "oncology",
                "expected_terms": ["tumor", "chemotherapy", "radiation"]
            },
            {
                "query": "heart failure patient with atrial fibrillation",
                "relevant_topic": "cardiovascular",
                "expected_terms": ["heart failure", "atrial fibrillation"]
            },
            {
                "query": "Type 2 diabetes with neuropathy complications",
                "relevant_topic": "diabetes",
                "expected_terms": ["diabetes", "neuropathy"]
            },
            {
                "query": "COPD exacerbation requiring oxygen therapy",
                "relevant_topic": "respiratory",
                "expected_terms": ["COPD", "oxygen"]
            },
            {
                "query": "stroke patient with motor deficits",
                "relevant_topic": "neurological",
                "expected_terms": ["stroke", "motor"]
            },
            {
                "query": "breast cancer with metastasis requiring targeted therapy",
                "relevant_topic": "oncology",
                "expected_terms": ["breast cancer", "metastasis", "targeted"]
            }
        ]
        
        return noisy_docs, queries
    
    def evaluate_retrieval(self, retrieved_docs: List[Document], query_info: Dict, top_k: int = 10) -> Dict[str, float]:
        """Evaluate retrieval quality."""
        relevant_topic = query_info["relevant_topic"]
        expected_terms = query_info["expected_terms"]
        
        # Calculate metrics
        relevant_in_top_k = 0
        first_relevant_rank = None
        relevance_scores = []
        
        for i, doc in enumerate(retrieved_docs[:top_k]):
            # Check if document is from relevant topic
            is_relevant = doc.metadata.custom_fields.get("topic") == relevant_topic
            
            # Check for expected terms (accounting for noise)
            term_matches = sum(1 for term in expected_terms 
                             if term.lower() in doc.content.lower() or 
                                self._fuzzy_match(term, doc.content))
            
            relevance_score = (1.0 if is_relevant else 0.0) + (0.2 * term_matches / len(expected_terms))
            relevance_scores.append(min(relevance_score, 1.0))
            
            if is_relevant:
                relevant_in_top_k += 1
                if first_relevant_rank is None:
                    first_relevant_rank = i + 1
        
        # Calculate metrics
        precision_at_k = relevant_in_top_k / top_k
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        return {
            "precision_at_k": precision_at_k,
            "mrr": mrr,
            "avg_relevance": np.mean(relevance_scores),
            "relevant_count": relevant_in_top_k
        }
    
    def _fuzzy_match(self, term: str, text: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for terms affected by noise."""
        term_lower = term.lower()
        text_lower = text.lower()
        
        # Check for partial matches
        if term_lower in text_lower:
            return True
        
        # Check for common OCR/typo variations
        variations = {
            'a': ['e', 'o'], 'e': ['a', 'c'], 'i': ['l', '1'],
            'o': ['0', 'e'], 'l': ['1', 'i'], 's': ['5', 'z']
        }
        
        for char, replacements in variations.items():
            if char in term_lower:
                for rep in replacements:
                    variant = term_lower.replace(char, rep)
                    if variant in text_lower:
                        return True
        
        return False
    
    def run_comparison_test(self, noise_level: str = "medium", top_k: int = 10) -> Dict[str, List[TestResult]]:
        """Run comprehensive comparison between classical and quantum-enhanced RAG."""
        print(f"\n{'='*60}")
        print(f"Medical RAG Test - Noise Level: {noise_level}")
        print(f"{'='*60}\n")
        
        # Prepare test data
        print("Generating medical documents...")
        docs, queries = self.prepare_test_data(docs_per_topic=20, noise_level=noise_level)
        print(f"Created {len(docs)} noisy medical documents")
        
        # Add documents to retriever
        print("\nIndexing documents...")
        self.retriever.add_documents(docs)
        
        # Test both methods
        results = {
            "classical": [],
            "quantum_enhanced": []
        }
        
        print("\nRunning retrieval tests...")
        print("-" * 60)
        
        for i, query_info in enumerate(queries):
            query = query_info["query"]
            print(f"\nQuery {i+1}: {query}")
            print(f"Expected topic: {query_info['relevant_topic']}")
            
            # Classical retrieval (FAISS only)
            # Get embeddings and perform direct FAISS search
            query_embedding = self.retriever.embedding_processor.encode_single_text(query)
            
            start_time = time.time()
            faiss_search_results = self.retriever.faiss_store.search(query_embedding, k=top_k)
            
            # Convert search results to documents
            classical_results = []
            for result in faiss_search_results:
                doc = self.retriever.document_store.get_document(result.doc_id)
                if doc:
                    classical_results.append(doc)
            
            classical_latency = (time.time() - start_time) * 1000
            classical_eval = self.evaluate_retrieval(classical_results, query_info, top_k)
            
            # Quantum-enhanced retrieval (two-stage with quantum reranking)
            start_time = time.time()
            quantum_retrieval_results = self.retriever.retrieve(query, k=top_k)
            
            # Convert retrieval results to documents
            quantum_results = []
            for result in quantum_retrieval_results:
                doc = self.retriever.document_store.get_document(result.doc_id)
                if doc:
                    quantum_results.append(doc)
                    
            quantum_latency = (time.time() - start_time) * 1000
            quantum_eval = self.evaluate_retrieval(quantum_results, query_info, top_k)
            
            # Store results
            results["classical"].append(TestResult(
                method="classical",
                query=query,
                top_k_accuracy=classical_eval["precision_at_k"],
                mrr=classical_eval["mrr"],
                latency_ms=classical_latency,
                relevance_scores=[classical_eval["avg_relevance"]]
            ))
            
            results["quantum_enhanced"].append(TestResult(
                method="quantum_enhanced",
                query=query,
                top_k_accuracy=quantum_eval["precision_at_k"],
                mrr=quantum_eval["mrr"],
                latency_ms=quantum_latency,
                relevance_scores=[quantum_eval["avg_relevance"]]
            ))
            
            # Print comparison
            print(f"  Classical - Precision@{top_k}: {classical_eval['precision_at_k']:.3f}, "
                  f"MRR: {classical_eval['mrr']:.3f}, Latency: {classical_latency:.1f}ms")
            print(f"  Quantum   - Precision@{top_k}: {quantum_eval['precision_at_k']:.3f}, "
                  f"MRR: {quantum_eval['mrr']:.3f}, Latency: {quantum_latency:.1f}ms")
            
            improvement = ((quantum_eval["precision_at_k"] - classical_eval["precision_at_k"]) / 
                          max(classical_eval["precision_at_k"], 0.001)) * 100
            print(f"  Improvement: {improvement:+.1f}%")
        
        return results
    
    def print_summary(self, results: Dict[str, List[TestResult]]):
        """Print comprehensive test summary."""
        print(f"\n{'='*60}")
        print("SUMMARY RESULTS")
        print(f"{'='*60}\n")
        
        for method in ["classical", "quantum_enhanced"]:
            method_results = results[method]
            
            avg_precision = np.mean([r.top_k_accuracy for r in method_results])
            avg_mrr = np.mean([r.mrr for r in method_results])
            avg_latency = np.mean([r.latency_ms for r in method_results])
            avg_relevance = np.mean([r.relevance_scores[0] for r in method_results])
            
            print(f"{method.upper()}:")
            print(f"  Average Precision@K: {avg_precision:.3f}")
            print(f"  Average MRR: {avg_mrr:.3f}")
            print(f"  Average Relevance: {avg_relevance:.3f}")
            print(f"  Average Latency: {avg_latency:.1f}ms")
            print()
        
        # Calculate improvements
        classical_precision = np.mean([r.top_k_accuracy for r in results["classical"]])
        quantum_precision = np.mean([r.top_k_accuracy for r in results["quantum_enhanced"]])
        
        classical_mrr = np.mean([r.mrr for r in results["classical"]])
        quantum_mrr = np.mean([r.mrr for r in results["quantum_enhanced"]])
        
        precision_improvement = ((quantum_precision - classical_precision) / 
                               max(classical_precision, 0.001)) * 100
        mrr_improvement = ((quantum_mrr - classical_mrr) / 
                          max(classical_mrr, 0.001)) * 100
        
        print("IMPROVEMENTS (Quantum vs Classical):")
        print(f"  Precision: {precision_improvement:+.1f}%")
        print(f"  MRR: {mrr_improvement:+.1f}%")
        
        # Latency comparison
        classical_latency = np.mean([r.latency_ms for r in results["classical"]])
        quantum_latency = np.mean([r.latency_ms for r in results["quantum_enhanced"]])
        latency_overhead = ((quantum_latency - classical_latency) / classical_latency) * 100
        
        print(f"  Latency Overhead: {latency_overhead:+.1f}%")
        print(f"\n{'='*60}\n")


def main():
    """Run the medical RAG comparison test."""
    tester = MedicalRAGTester()
    
    # Test with different noise levels
    noise_levels = ["low", "medium", "high"]
    all_results = {}
    
    for noise_level in noise_levels:
        results = tester.run_comparison_test(noise_level=noise_level, top_k=10)
        all_results[noise_level] = results
        tester.print_summary(results)
    
    # Final comparison across noise levels
    print("\nFINAL COMPARISON ACROSS NOISE LEVELS:")
    print("="*60)
    
    for noise_level in noise_levels:
        results = all_results[noise_level]
        classical_precision = np.mean([r.top_k_accuracy for r in results["classical"]])
        quantum_precision = np.mean([r.top_k_accuracy for r in results["quantum_enhanced"]])
        improvement = ((quantum_precision - classical_precision) / 
                      max(classical_precision, 0.001)) * 100
        
        print(f"{noise_level.upper()} noise - Quantum improvement: {improvement:+.1f}%")
    
    # Save detailed results
    with open("medical_rag_test_results.json", "w") as f:
        json_results = {}
        for noise_level, results in all_results.items():
            json_results[noise_level] = {
                method: [
                    {
                        "query": r.query,
                        "precision": r.top_k_accuracy,
                        "mrr": r.mrr,
                        "latency_ms": r.latency_ms
                    }
                    for r in method_results
                ]
                for method, method_results in results.items()
            }
        json.dump(json_results, f, indent=2)
    
    print("\nDetailed results saved to medical_rag_test_results.json")


if __name__ == "__main__":
    main()