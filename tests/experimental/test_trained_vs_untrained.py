#!/usr/bin/env python3
"""
Compare reranking performance between untrained and trained QPMeL models.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_rerank.core.rag_reranker import QuantumRAGReranker
from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig

def create_test_scenarios() -> List[Dict]:
    """Create test scenarios for comparing reranking performance."""
    return [
        {
            "name": "Quantum Computing Query",
            "query": "What is quantum computing and how does it work?",
            "documents": [
                # Highly relevant
                "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information in fundamentally new ways, potentially solving certain problems exponentially faster than classical computers.",
                "Quantum computers leverage qubits that can exist in superposition states, allowing them to perform multiple calculations simultaneously through quantum parallelism and interference.",
                
                # Moderately relevant
                "Classical computers use binary bits for computation, while quantum systems use quantum mechanical phenomena to achieve computational advantages in specific problem domains.",
                "The field of quantum information science combines quantum mechanics with computer science to develop new computational paradigms and communication protocols.",
                
                # Less relevant
                "Computer programming involves writing instructions for machines to execute, using languages like Python, Java, and C++ to create software applications.",
                "Data science uses statistical methods and machine learning to extract insights from large datasets across various industries and applications.",
                
                # Irrelevant
                "Cooking pasta requires boiling water, adding salt, and cooking for the time specified on the package until al dente texture is achieved.",
                "Weather patterns are influenced by atmospheric pressure, temperature gradients, and ocean currents that create complex meteorological systems."
            ],
            "expected_relevant": [0, 1, 2, 3]  # Indices of relevant documents
        },
        
        {
            "name": "Machine Learning Query", 
            "query": "How do neural networks learn and improve their performance?",
            "documents": [
                # Highly relevant
                "Neural networks learn through backpropagation, adjusting weights and biases based on error gradients to minimize loss functions and improve prediction accuracy.",
                "Deep learning networks use multiple layers to extract hierarchical features, with each layer learning increasingly complex representations through supervised training.",
                
                # Moderately relevant
                "Machine learning algorithms improve performance through training on labeled datasets, using optimization techniques to find optimal model parameters.",
                "Artificial intelligence systems can adapt and learn from experience, using various learning paradigms including supervised, unsupervised, and reinforcement learning.",
                
                # Less relevant
                "Computer networks connect devices to share information and resources, using protocols like TCP/IP to ensure reliable data transmission across the internet.",
                "Database systems store and organize information efficiently, providing fast access and retrieval capabilities for large-scale applications.",
                
                # Irrelevant
                "Gardening requires understanding soil conditions, plant needs, and seasonal cycles to successfully grow flowers, vegetables, and other plants.",
                "Photography involves controlling light, composition, and camera settings to capture compelling images that tell stories or convey emotions."
            ],
            "expected_relevant": [0, 1, 2, 3]
        },
        
        {
            "name": "Scientific Research Query",
            "query": "What are the latest advances in renewable energy technology?",
            "documents": [
                # Highly relevant
                "Recent advances in solar panel efficiency include perovskite tandem cells achieving over 30% efficiency and new manufacturing techniques reducing costs significantly.",
                "Wind energy innovations focus on larger turbines with advanced blade designs and offshore installations that can capture stronger, more consistent winds.",
                
                # Moderately relevant
                "Energy storage breakthroughs include improved battery technologies and grid-scale solutions that enable better integration of renewable sources into power systems.",
                "Smart grid technologies help optimize renewable energy distribution and consumption, reducing waste and improving overall energy system efficiency.",
                
                # Less relevant
                "Environmental science studies ecosystems and their interactions, helping us understand climate change impacts and develop sustainability strategies.",
                "Green building practices incorporate energy-efficient designs and sustainable materials to reduce environmental impact and operating costs.",
                
                # Irrelevant
                "Fashion trends evolve seasonally, influenced by cultural movements, celebrity styles, and designer innovations that shape consumer preferences.",
                "Sports psychology helps athletes improve mental performance through visualization, goal setting, and stress management techniques."
            ],
            "expected_relevant": [0, 1, 2, 3]
        }
    ]

def test_reranker(reranker: QuantumRAGReranker, scenario: Dict, model_name: str) -> Dict:
    """Test a reranker on a scenario and return metrics."""
    query = scenario["query"]
    documents = scenario["documents"]
    expected_relevant = set(scenario["expected_relevant"])
    
    # Test different methods
    methods = ["classical", "quantum", "hybrid"]
    results = {}
    
    for method in methods:
        start_time = time.time()
        try:
            ranked_docs = reranker.rerank(query, documents, method=method, top_k=len(documents))
            end_time = time.time()
            
            # Calculate metrics
            relevance_scores = []
            for i, result in enumerate(ranked_docs):
                # Find original document index
                doc_text = result.get('text', result.get('content', ''))
                original_idx = None
                for j, original_doc in enumerate(documents):
                    if original_doc in doc_text or doc_text in original_doc:
                        original_idx = j
                        break
                
                # Calculate relevance score (1 if relevant, 0 if not)
                relevance = 1.0 if original_idx in expected_relevant else 0.0
                relevance_scores.append(relevance)
            
            # Calculate ranking metrics
            top_3_relevant = sum(relevance_scores[:3])
            top_5_relevant = sum(relevance_scores[:5]) if len(relevance_scores) >= 5 else sum(relevance_scores)
            ndcg_3 = calculate_ndcg(relevance_scores[:3])
            ndcg_5 = calculate_ndcg(relevance_scores[:5]) if len(relevance_scores) >= 5 else calculate_ndcg(relevance_scores)
            
            results[method] = {
                "top_3_relevant": top_3_relevant,
                "top_5_relevant": top_5_relevant,
                "ndcg_3": ndcg_3,
                "ndcg_5": ndcg_5,
                "latency_ms": (end_time - start_time) * 1000,
                "total_relevant": len(expected_relevant)
            }
            
        except Exception as e:
            print(f"  ❌ {method} method failed: {e}")
            results[method] = None
    
    return results

def calculate_ndcg(relevance_scores: List[float]) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    if not relevance_scores:
        return 0.0
    
    # DCG calculation
    dcg = relevance_scores[0]
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / (i + 1)
    
    # IDCG calculation (perfect ranking)
    sorted_relevance = sorted(relevance_scores, reverse=True)
    idcg = sorted_relevance[0] if sorted_relevance else 0
    for i in range(1, len(sorted_relevance)):
        idcg += sorted_relevance[i] / (i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0

def create_trained_reranker(model_path: str) -> QuantumRAGReranker:
    """Create a reranker using trained QPMeL parameters."""
    config = QPMeLTrainingConfig(
        qpmel_config=QPMeLConfig(n_qubits=2, n_layers=1),
        batch_size=8
    )
    trainer = QPMeLTrainer(config=config)
    trainer.load_model(model_path)
    return trainer.get_trained_reranker()

def main():
    """Main comparison function."""
    print("🔬 Trained vs Untrained QPMeL Reranking Comparison")
    print("=" * 60)
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Initialize rerankers
    print("\n🚀 Initializing Rerankers...")
    
    # Untrained (default random parameters)
    untrained_reranker = QuantumRAGReranker()
    print("✅ Untrained reranker (random parameters)")
    
    # Trained models
    trained_rerankers = {}
    
    model_files = [
        ("models/qpmel_mvp.pt", "MVP (500 triplets, 5 epochs)"),
        ("models/qpmel_extended.pt", "Extended (2000 triplets, 6+ epochs)")
    ]
    
    for model_path, model_name in model_files:
        try:
            trained_reranker = create_trained_reranker(model_path)
            trained_rerankers[model_name] = trained_reranker
            print(f"✅ {model_name}")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
    
    # Run comparison tests
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n📊 Testing Scenario: {scenario['name']}")
        print(f"Query: {scenario['query'][:80]}...")
        print(f"Documents: {len(scenario['documents'])} ({len(scenario['expected_relevant'])} relevant)")
        
        scenario_results = {}
        
        # Test untrained reranker
        print("\n  🔍 Testing Untrained Reranker...")
        untrained_results = test_reranker(untrained_reranker, scenario, "Untrained")
        scenario_results["Untrained"] = untrained_results
        
        # Test trained rerankers
        for model_name, reranker in trained_rerankers.items():
            print(f"\n  🧠 Testing {model_name}...")
            trained_results = test_reranker(reranker, scenario, model_name)
            scenario_results[model_name] = trained_results
        
        all_results[scenario['name']] = scenario_results
    
    # Generate comparison report
    print(f"\n📈 Performance Comparison Report")
    print("=" * 60)
    
    for scenario_name, scenario_results in all_results.items():
        print(f"\n🎯 {scenario_name}")
        print("-" * 40)
        
        # Print method comparison table
        methods = ["classical", "quantum", "hybrid"]
        
        for method in methods:
            print(f"\n  📋 {method.upper()} Method:")
            print(f"    {'Model':<25} {'Top-3 Rel':<10} {'NDCG@3':<8} {'Latency':<10}")
            print(f"    {'-'*25} {'-'*10} {'-'*8} {'-'*10}")
            
            for model_name, results in scenario_results.items():
                if results and results.get(method):
                    metrics = results[method]
                    top3 = metrics['top_3_relevant']
                    ndcg3 = metrics['ndcg_3']
                    latency = metrics['latency_ms']
                    print(f"    {model_name:<25} {top3:<10.1f} {ndcg3:<8.3f} {latency:<10.1f}ms")
                else:
                    print(f"    {model_name:<25} {'FAILED':<10} {'N/A':<8} {'N/A':<10}")
    
    # Calculate overall improvements
    print(f"\n🏆 Overall Performance Summary")
    print("=" * 60)
    
    # Aggregate metrics across all scenarios and methods
    untrained_metrics = []
    trained_metrics = {"MVP (500 triplets, 5 epochs)": [], "Extended (2000 triplets, 6+ epochs)": []}
    
    for scenario_results in all_results.values():
        for method in ["classical", "quantum", "hybrid"]:
            if "Untrained" in scenario_results and scenario_results["Untrained"].get(method):
                untrained_metrics.append(scenario_results["Untrained"][method]['ndcg_3'])
            
            for model_name in trained_metrics.keys():
                if model_name in scenario_results and scenario_results[model_name].get(method):
                    trained_metrics[model_name].append(scenario_results[model_name][method]['ndcg_3'])
    
    if untrained_metrics:
        avg_untrained = sum(untrained_metrics) / len(untrained_metrics)
        print(f"📊 Average NDCG@3 (Untrained): {avg_untrained:.3f}")
        
        for model_name, metrics in trained_metrics.items():
            if metrics:
                avg_trained = sum(metrics) / len(metrics)
                improvement = ((avg_trained - avg_untrained) / avg_untrained * 100) if avg_untrained > 0 else 0
                print(f"📊 Average NDCG@3 ({model_name}): {avg_trained:.3f} ({improvement:+.1f}%)")
    
    # Save detailed results
    with open('comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Detailed results saved to comparison_results.json")
    
    print(f"\n✨ Comparison Complete!")

if __name__ == "__main__":
    main()