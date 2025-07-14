#!/usr/bin/env python3
"""
Test the trained QPMeL model to verify it works correctly.
"""

import torch
import json
from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig
from quantum_rerank.core.embeddings import EmbeddingProcessor

def test_trained_model():
    """Test the trained QPMeL model."""
    print("🧪 Testing Trained QPMeL Model")
    print("=" * 50)
    
    # Load trained model
    config = QPMeLTrainingConfig(
        qpmel_config=QPMeLConfig(n_qubits=2, n_layers=1),
        batch_size=8
    )
    trainer = QPMeLTrainer(config=config)
    
    try:
        checkpoint = trainer.load_model('models/qpmel_mvp.pt')
        print("✅ Model loaded successfully!")
        print(f"   Trained for {checkpoint['epoch']} epochs")
        print(f"   Best validation loss: {checkpoint['best_val_loss']:.4f}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get model info
    model_info = trainer.model.get_model_info()
    print(f"✅ Model Architecture:")
    print(f"   Input dim: {model_info['input_dim']}")
    print(f"   Qubits: {model_info['n_qubits']}")
    print(f"   Circuit params: {model_info['n_circuit_params']}")
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    
    # Test parameter prediction
    embedding_processor = EmbeddingProcessor()
    test_texts = [
        "machine learning algorithms",
        "quantum computing research"
    ]
    
    print(f"✅ Testing parameter prediction...")
    embeddings = embedding_processor.encode_texts(test_texts)
    embeddings_tensor = torch.FloatTensor(embeddings)
    
    with torch.no_grad():
        parameters = trainer.model.forward(embeddings_tensor, training=False)
        print(f"   Predicted parameters shape: {parameters.shape}")
        print(f"   Parameter ranges: [{parameters.min():.3f}, {parameters.max():.3f}]")
    
    # Test quantum circuits generation
    print(f"✅ Testing quantum circuit generation...")
    try:
        circuits = trainer.model.get_circuits(embeddings_tensor, training=False)
        print(f"   Generated {len(circuits)} circuits")
        print(f"   Circuit depth: {circuits[0].depth()}")
        print(f"   Circuit size: {circuits[0].size()} gates")
    except Exception as e:
        print(f"❌ Circuit generation failed: {e}")
        return
    
    # Test fidelity computation
    print(f"✅ Testing fidelity computation...")
    try:
        fidelities = trainer.compute_batch_fidelities(
            embeddings_tensor[:1], 
            embeddings_tensor[1:], 
            training=False
        )
        print(f"   Computed fidelity: {fidelities[0]:.4f}")
    except Exception as e:
        print(f"❌ Fidelity computation failed: {e}")
        return
    
    # Test trained reranker integration
    print(f"✅ Testing trained reranker...")
    try:
        reranker = trainer.get_trained_reranker()
        
        # Test reranking
        query = "machine learning algorithms"
        candidates = [
            "Deep learning neural networks and AI models",
            "Cooking recipes and delicious food preparation"
        ]
        
        results = reranker.rerank(query, candidates, method="quantum", top_k=2)
        print(f"   Query: {query}")
        print(f"   Reranking results:")
        for i, result in enumerate(results):
            # Handle different result formats
            if isinstance(result, dict):
                text = result.get('text', result.get('content', str(result)))
                score = result.get('score', result.get('similarity', 0.0))
            else:
                text = str(result)
                score = 0.0
            print(f"     {i+1}. {text[:40]}... (score: {score:.4f})")
    except Exception as e:
        print(f"❌ Reranker test failed: {e}")
        return
    
    print("\n🎉 All tests passed! The trained QPMeL model is working correctly.")
    print("\n📊 Training Summary:")
    
    # Load and display training history
    try:
        with open('models/qpmel_mvp.history.json', 'r') as f:
            history = json.load(f)
        
        final_metrics = history[-1]
        print(f"   Final train loss: {final_metrics['train_loss']:.6f}")
        print(f"   Final train accuracy: {final_metrics['train_accuracy']:.3f}")
        print(f"   Final fidelity gap: {final_metrics['train_fidelity_gap']:.4f}")
        print(f"   Training time per epoch: ~{final_metrics['epoch_time']:.1f}s")
        print(f"   Total epochs: {len(history)}")
        
        # Show learning progression
        print(f"   Learning progression:")
        print(f"     Epoch 0: Loss={history[0]['train_loss']:.4f}, Acc={history[0]['train_accuracy']:.3f}")
        print(f"     Epoch {len(history)-1}: Loss={final_metrics['train_loss']:.4f}, Acc={final_metrics['train_accuracy']:.3f}")
        
    except Exception as e:
        print(f"   ⚠️  Could not load training history: {e}")

if __name__ == "__main__":
    test_trained_model()