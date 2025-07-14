#!/usr/bin/env python3
"""
QPMeL Evaluation Script

Compare trained QPMeL model against baseline approaches to measure
the effectiveness of quantum triplet loss training.

Usage:
    python evaluate_qpmel.py --model models/qpmel_trained.pt --dataset nfcorpus
    python evaluate_qpmel.py --model models/qpmel_trained.pt --compare-all
"""

import argparse
import logging
import sys
import torch
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.training.triplet_generator import (
    create_synthetic_triplets, load_nfcorpus_triplets, 
    load_msmarco_triplets, load_sentence_transformers_triplets
)
from quantum_rerank.core.rag_reranker import QuantumRAGReranker
from quantum_rerank.core.embeddings import EmbeddingProcessor
from quantum_rerank.utils.logging_config import setup_logging

class QPMeLEvaluator:
    """
    Comprehensive evaluator for QPMeL trained models.
    
    Compares trained quantum models against:
    - Random baseline
    - Classical cosine similarity
    - Untrained quantum model
    - Original quantum reranker
    """
    
    def __init__(self):
        self.embedding_processor = EmbeddingProcessor()
        self.results = {}
        
    def load_trained_model(self, model_path: str) -> QPMeLTrainer:
        """Load a trained QPMeL model."""
        logger = logging.getLogger(__name__)
        
        # Create trainer and load model
        trainer = QPMeLTrainer(
            config=QPMeLTrainingConfig(),  # Will be overridden by loaded config
            embedding_processor=self.embedding_processor
        )
        
        checkpoint = trainer.load_model(model_path)
        logger.info(f"Loaded trained model from {model_path}")
        logger.info(f"Model info: {checkpoint.get('model_info', {})}")
        
        return trainer
    
    def create_baseline_rerankers(self):
        """Create baseline reranking methods for comparison."""
        
        def random_reranker(query: str, documents: List[str]) -> List[Dict]:
            """Random baseline."""
            import random
            docs_with_scores = [(doc, random.random()) for doc in documents]
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            return [{"text": doc, "similarity_score": score, "method": "random"} 
                   for doc, score in docs_with_scores]
        
        def classical_reranker(query: str, documents: List[str]) -> List[Dict]:
            """Classical cosine similarity baseline."""
            all_texts = [query] + documents
            embeddings = self.embedding_processor.encode_texts(all_texts)
            
            query_embedding = embeddings[0]
            doc_embeddings = embeddings[1:]
            
            similarities = []
            for doc, doc_emb in zip(documents, doc_embeddings):
                sim = self.embedding_processor.compute_classical_similarity(
                    query_embedding, doc_emb
                )
                similarities.append((doc, float(sim)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [{"text": doc, "similarity_score": score, "method": "classical"} 
                   for doc, score in similarities]
        
        def untrained_quantum_reranker(query: str, documents: List[str]) -> List[Dict]:
            """Untrained quantum model baseline."""
            # Create fresh quantum reranker (untrained parameters)
            reranker = QuantumRAGReranker()
            return reranker.rerank(query, documents, method="quantum", top_k=len(documents))
        
        return {
            "Random": random_reranker,
            "Classical-Cosine": classical_reranker,
            "Quantum-Untrained": untrained_quantum_reranker
        }
    
    def evaluate_triplet_accuracy(self, 
                                 rerank_func,
                                 triplets: List[Tuple[str, str, str]],
                                 method_name: str) -> Dict[str, Any]:
        """
        Evaluate reranker on triplet accuracy.
        
        Accuracy = percentage of triplets where positive document 
        is ranked higher than negative document.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Evaluating {method_name} on {len(triplets)} triplets")
        
        correct_rankings = 0
        total_tests = 0
        similarities_pos = []
        similarities_neg = []
        processing_times = []
        
        for i, (anchor, positive, negative) in enumerate(triplets):
            try:
                start_time = time.time()
                
                # Rerank positive and negative documents
                candidates = [positive, negative]
                results = rerank_func(anchor, candidates)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time * 1000)  # Convert to ms
                
                if len(results) >= 2:
                    # Check if positive is ranked higher than negative
                    top_doc = results[0]["text"]
                    top_score = results[0]["similarity_score"]
                    second_score = results[1]["similarity_score"]
                    
                    if top_doc == positive:
                        correct_rankings += 1
                        similarities_pos.append(top_score)
                        similarities_neg.append(second_score)
                    else:
                        similarities_pos.append(second_score)
                        similarities_neg.append(top_score)
                    
                    total_tests += 1
                
            except Exception as e:
                logger.debug(f"Failed on triplet {i}: {e}")
                continue
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(triplets)} triplets")
        
        if total_tests == 0:
            return {"error": "No successful evaluations"}
        
        accuracy = correct_rankings / total_tests
        avg_sim_pos = np.mean(similarities_pos) if similarities_pos else 0
        avg_sim_neg = np.mean(similarities_neg) if similarities_neg else 0
        similarity_gap = avg_sim_pos - avg_sim_neg
        avg_time = np.mean(processing_times) if processing_times else 0
        
        results = {
            "method": method_name,
            "accuracy": accuracy,
            "correct_rankings": correct_rankings,
            "total_tests": total_tests,
            "avg_similarity_positive": avg_sim_pos,
            "avg_similarity_negative": avg_sim_neg,
            "similarity_gap": similarity_gap,
            "avg_processing_time_ms": avg_time,
            "std_processing_time_ms": np.std(processing_times) if processing_times else 0
        }
        
        logger.info(f"{method_name} Results: "
                   f"Accuracy={accuracy:.3f}, "
                   f"Gap={similarity_gap:.4f}, "
                   f"Time={avg_time:.1f}ms")
        
        return results
    
    def evaluate_ranking_quality(self,
                                rerank_func,
                                test_scenarios: List[Dict],
                                method_name: str) -> Dict[str, Any]:
        """
        Evaluate ranking quality on test scenarios with known relevance.
        
        Uses NDCG and other ranking metrics.
        """
        logger = logging.getLogger(__name__)
        
        ndcg_scores = []
        mrr_scores = []
        
        for scenario in test_scenarios:
            query = scenario["query"]
            documents = scenario["documents"]
            expected_ranking = scenario["expected_ranking"]
            
            try:
                # Get reranking results
                results = rerank_func(query, documents)
                
                # Calculate NDCG
                ndcg = self._calculate_ndcg(results, documents, expected_ranking)
                ndcg_scores.append(ndcg)
                
                # Calculate MRR (Mean Reciprocal Rank)
                mrr = self._calculate_mrr(results, documents, expected_ranking)
                mrr_scores.append(mrr)
                
            except Exception as e:
                logger.debug(f"Failed on scenario: {e}")
                continue
        
        return {
            "method": method_name,
            "avg_ndcg": np.mean(ndcg_scores) if ndcg_scores else 0,
            "avg_mrr": np.mean(mrr_scores) if mrr_scores else 0,
            "num_scenarios": len(ndcg_scores)
        }
    
    def _calculate_ndcg(self, results: List[Dict], documents: List[str], expected_ranking: List[int], k: int = 10) -> float:
        """Calculate NDCG@k score."""
        import math
        
        # Map results back to document indices
        actual_ranking = []
        for result in results[:k]:
            result_text = result["text"]
            for i, doc in enumerate(documents):
                if doc == result_text:
                    actual_ranking.append(i)
                    break
        
        # Create relevance scores (higher rank = higher relevance)
        relevance_scores = {}
        for pos, doc_idx in enumerate(expected_ranking):
            relevance_scores[doc_idx] = len(expected_ranking) - pos
        
        # Calculate DCG
        dcg = 0.0
        for pos, doc_idx in enumerate(actual_ranking):
            relevance = relevance_scores.get(doc_idx, 0)
            dcg += relevance / math.log2(pos + 2)
        
        # Calculate IDCG
        idcg = 0.0
        for pos, doc_idx in enumerate(expected_ranking[:k]):
            relevance = relevance_scores.get(doc_idx, 0)
            idcg += relevance / math.log2(pos + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, results: List[Dict], documents: List[str], expected_ranking: List[int]) -> float:
        """Calculate Mean Reciprocal Rank."""
        # Find first relevant document in results
        for pos, result in enumerate(results):
            result_text = result["text"]
            for doc_idx, doc in enumerate(documents):
                if doc == result_text and doc_idx in expected_ranking[:3]:  # Top 3 are relevant
                    return 1.0 / (pos + 1)
        return 0.0
    
    def run_comprehensive_evaluation(self, 
                                   trained_model_path: str,
                                   test_triplets: List[Tuple[str, str, str]],
                                   test_scenarios: List[Dict] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation comparing all methods.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting comprehensive QPMeL evaluation")
        
        # Load trained model
        trained_trainer = self.load_trained_model(trained_model_path)
        trained_reranker = trained_trainer.get_trained_reranker()
        
        def trained_qpmel_reranker(query: str, documents: List[str]) -> List[Dict]:
            return trained_reranker.rerank(query, documents, method="quantum", top_k=len(documents))
        
        # Get all reranking methods
        baseline_methods = self.create_baseline_rerankers()
        all_methods = {
            **baseline_methods,
            "QPMeL-Trained": trained_qpmel_reranker
        }
        
        # Evaluate on triplet accuracy
        logger.info("Evaluating triplet accuracy...")
        triplet_results = {}
        for method_name, rerank_func in all_methods.items():
            triplet_results[method_name] = self.evaluate_triplet_accuracy(
                rerank_func, test_triplets, method_name
            )
        
        # Evaluate on ranking quality (if test scenarios provided)
        ranking_results = {}
        if test_scenarios:
            logger.info("Evaluating ranking quality...")
            for method_name, rerank_func in all_methods.items():
                ranking_results[method_name] = self.evaluate_ranking_quality(
                    rerank_func, test_scenarios, method_name
                )
        
        # Compile comprehensive results
        results = {
            "evaluation_summary": {
                "num_triplets": len(test_triplets),
                "num_scenarios": len(test_scenarios) if test_scenarios else 0,
                "methods_compared": list(all_methods.keys())
            },
            "triplet_accuracy": triplet_results,
            "ranking_quality": ranking_results,
            "model_info": trained_trainer.model.get_model_info(),
            "training_history": trained_trainer.training_history[-5:] if trained_trainer.training_history else []  # Last 5 epochs
        }
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of evaluation results."""
        logger = logging.getLogger(__name__)
        
        print("\n" + "="*80)
        print("📊 QPMeL EVALUATION SUMMARY")
        print("="*80)
        
        # Triplet accuracy results
        triplet_results = results["triplet_accuracy"]
        if triplet_results:
            print("\n🎯 TRIPLET ACCURACY COMPARISON")
            print("-" * 50)
            print(f"{'Method':<20} {'Accuracy':<10} {'Sim Gap':<10} {'Time(ms)':<10}")
            print("-" * 50)
            
            # Sort by accuracy
            sorted_methods = sorted(triplet_results.items(), 
                                  key=lambda x: x[1].get("accuracy", 0), reverse=True)
            
            for method_name, metrics in sorted_methods:
                if "error" in metrics:
                    continue
                    
                accuracy = metrics["accuracy"]
                sim_gap = metrics["similarity_gap"]
                avg_time = metrics["avg_processing_time_ms"]
                
                print(f"{method_name:<20} {accuracy:<10.3f} {sim_gap:<10.4f} {avg_time:<10.1f}")
            
            # Highlight best method
            best_method = sorted_methods[0]
            print(f"\n🏆 Best Method: {best_method[0]} (Accuracy: {best_method[1]['accuracy']:.3f})")
            
            # QPMeL vs Classical comparison
            qpmel_results = triplet_results.get("QPMeL-Trained", {})
            classical_results = triplet_results.get("Classical-Cosine", {})
            
            if qpmel_results and classical_results and "error" not in qpmel_results:
                qpmel_acc = qpmel_results["accuracy"]
                classical_acc = classical_results["accuracy"]
                improvement = ((qpmel_acc - classical_acc) / classical_acc) * 100
                
                print(f"\n🔬 QPMeL vs Classical Analysis:")
                print(f"   QPMeL Accuracy: {qpmel_acc:.3f}")
                print(f"   Classical Accuracy: {classical_acc:.3f}")
                print(f"   Improvement: {improvement:+.2f}%")
                
                if improvement > 5:
                    print("   ✅ QPMeL shows significant improvement!")
                elif improvement > 0:
                    print("   🟡 QPMeL shows modest improvement")
                else:
                    print("   ⚠️  Classical approach performs better")
        
        # Ranking quality results
        ranking_results = results["ranking_quality"]
        if ranking_results:
            print("\n📊 RANKING QUALITY COMPARISON")
            print("-" * 40)
            print(f"{'Method':<20} {'NDCG':<10} {'MRR':<10}")
            print("-" * 40)
            
            for method_name, metrics in ranking_results.items():
                ndcg = metrics["avg_ndcg"]
                mrr = metrics["avg_mrr"]
                print(f"{method_name:<20} {ndcg:<10.4f} {mrr:<10.4f}")
        
        # Model information
        model_info = results.get("model_info", {})
        if model_info:
            print(f"\n🔧 MODEL ARCHITECTURE")
            print("-" * 30)
            print(f"   Qubits: {model_info.get('n_qubits', 'N/A')}")
            print(f"   Circuit Parameters: {model_info.get('n_circuit_params', 'N/A')}")
            print(f"   Total Parameters: {model_info.get('total_parameters', 'N/A')}")
            print(f"   QRC Enabled: {model_info.get('enable_qrc', 'N/A')}")
        
        print("\n" + "="*80)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained QPMeL model")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained QPMeL model')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'nfcorpus', 'msmarco', 'sentence-transformers'],
                       help='Test dataset')
    parser.add_argument('--num-triplets', type=int, default=500,
                       help='Number of test triplets')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare against all baseline methods')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Path to save evaluation results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    return parser.parse_args()

def load_test_data(dataset: str, num_triplets: int):
    """Load test triplets based on dataset."""
    if dataset == 'synthetic':
        return create_synthetic_triplets(num_triplets)
    elif dataset == 'nfcorpus':
        triplets = load_nfcorpus_triplets()
        return triplets[:num_triplets] if len(triplets) > num_triplets else triplets
    elif dataset == 'msmarco':
        triplets = load_msmarco_triplets()
        return triplets[:num_triplets] if len(triplets) > num_triplets else triplets
    elif dataset == 'sentence-transformers':
        triplets = load_sentence_transformers_triplets()
        return triplets[:num_triplets] if len(triplets) > num_triplets else triplets
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting QPMeL evaluation")
    
    try:
        # Load test data
        logger.info(f"Loading test dataset: {args.dataset}")
        test_triplets = load_test_data(args.dataset, args.num_triplets)
        logger.info(f"Loaded {len(test_triplets)} test triplets")
        
        # Create evaluator
        evaluator = QPMeLEvaluator()
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation(
            trained_model_path=args.model,
            test_triplets=test_triplets
        )
        
        # Print summary
        evaluator.print_evaluation_summary(results)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.save_results}")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()