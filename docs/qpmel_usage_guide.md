# QPMeL (Quantum Polar Metric Learning) Usage Guide

This guide shows you how to train and use QPMeL to improve your quantum reranker's semantic accuracy through quantum triplet loss training.

## 🎯 What is QPMeL?

QPMeL trains your quantum circuit parameters to actually understand semantic similarity, instead of using random parameters. It uses:
- **Polar encoding**: 2 angles per qubit (θ, γ) for efficient quantum state representation
- **Fidelity triplet loss**: Trains so quantum fidelity correlates with semantic similarity  
- **Shallow circuits**: Only 4 qubits, ~15 gates - practical for current hardware
- **End-to-end training**: Classical neural network + quantum circuit optimization

## 🚀 Quick Start

### 1. Train a QPMeL Model

```bash
# Train on synthetic data (quick test)
python train_qpmel.py --dataset synthetic --epochs 20 --batch-size 16

# Train on NFCorpus (real IR dataset) 
python train_qpmel.py --dataset nfcorpus --epochs 50 --batch-size 8

# Train on SentenceTransformers dataset
python train_qpmel.py --dataset sentence-transformers --epochs 30 --lr 0.001
```

### 2. Evaluate the Trained Model

```bash
# Evaluate against baselines
python evaluate_qpmel.py --model models/qpmel_trained.pt --dataset synthetic

# Compare all methods comprehensively
python evaluate_qpmel.py --model models/qpmel_trained.pt --compare-all --save-results results.json
```

### 3. Use the Trained Model

```python
from quantum_rerank.core.qpmel_integration import load_qpmel_reranker

# Load trained QPMeL reranker
reranker = load_qpmel_reranker("models/qpmel_trained.pt")

# Use it like any reranker
query = "What is quantum computing?"
documents = [
    "Quantum computing uses quantum mechanics for computation",
    "Classical computers use binary logic",
    "Weather forecast for tomorrow"
]

results = reranker.rerank(query, documents, method="quantum")
print(f"Best document: {results[0]['text']}")
```

## 📊 Training Options

### Dataset Options

| Dataset | Description | Training Time | Quality |
|---------|-------------|---------------|---------|
| `synthetic` | Generated test triplets | ~5 min | Good for testing |
| `nfcorpus` | Nutrition/medical IR dataset | ~30 min | High quality |
| `sentence-transformers` | AllNLI triplet dataset | ~20 min | Very high quality |
| `msmarco` | MS MARCO passage ranking | ~60 min | Production quality |

### Model Configuration

```bash
# Small model (fast training)
python train_qpmel.py --n-qubits 2 --hidden-dims 256 128 --batch-size 32

# Large model (better accuracy)  
python train_qpmel.py --n-qubits 4 --hidden-dims 512 256 128 --batch-size 8

# Enable Quantum Residual Correction for stability
python train_qpmel.py --enable-qrc --lr 0.0005
```

## 🔬 Integration with Existing System

### Replace Existing Reranker

```python
from quantum_rerank.core.rag_reranker import QuantumRAGReranker
from quantum_rerank.core.qpmel_integration import upgrade_existing_reranker

# Your existing reranker
old_reranker = QuantumRAGReranker()

# Upgrade with trained QPMeL parameters
new_reranker = upgrade_existing_reranker(old_reranker, "models/qpmel_trained.pt")

# Now new_reranker uses trained parameters!
```

### Use in Your Application

```python
from quantum_rerank.core.qpmel_integration import QPMeLReranker

class MyRAGApplication:
    def __init__(self):
        # Use QPMeL-enhanced reranker
        self.reranker = QPMeLReranker(qpmel_model_path="models/qpmel_trained.pt")
    
    def search(self, query: str, candidates: List[str]) -> List[str]:
        # Your existing retrieval logic
        initial_results = self.initial_retrieval(query)
        
        # Enhanced reranking with trained quantum parameters
        reranked = self.reranker.rerank(query, initial_results, method="quantum")
        
        return [r["text"] for r in reranked]
```

## 📈 Expected Improvements

Based on QPMeL paper results, you can expect:

- **3x better separation**: Quantum fidelity better distinguishes similar vs dissimilar texts
- **5-15% accuracy gain**: Over untrained quantum or classical-only approaches  
- **Faster circuits**: Half the gates and depth of previous quantum approaches
- **Stable training**: Quantum Residual Correction prevents gradient issues

## 🛠️ Advanced Usage

### Custom Triplet Data

```python
from quantum_rerank.training.triplet_generator import TripletGenerator

# Create custom triplets
generator = TripletGenerator()
triplets = generator.from_custom_data(
    queries=["quantum computing", "machine learning"],
    relevant_docs=[
        ["quantum algorithms", "qubits and gates"],  # For query 1
        ["neural networks", "deep learning"]        # For query 2  
    ],
    irrelevant_docs=["cooking recipes", "weather data"]
)

# Save for reuse
generator.save_triplets(triplets, "my_triplets.json")

# Train with custom data
python train_qpmel.py --dataset custom --triplets-file my_triplets.json
```

### Hyperparameter Tuning

```python
from quantum_rerank.training.qpmel_trainer import QPMeLTrainingConfig
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig

# Experiment with different configurations
configs = [
    QPMeLConfig(n_qubits=2, n_layers=1),  # Fast
    QPMeLConfig(n_qubits=4, n_layers=1),  # Balanced  
    QPMeLConfig(n_qubits=4, n_layers=2),  # Accurate
]

for config in configs:
    # Train and evaluate each configuration
    trainer = QPMeLTrainer(QPMeLTrainingConfig(qpmel_config=config))
    # ... training code
```

### Performance Analysis

```python
# Get detailed model information
info = reranker.get_model_info()
print(f"Circuit depth: {info['qpmel_info']['circuit_properties']['circuit_depth']}")
print(f"Total parameters: {info['qpmel_info']['total_parameters']}")

# Analyze training progress
checkpoint = torch.load("models/qpmel_trained.pt")
history = checkpoint['training_history']

import matplotlib.pyplot as plt
epochs = [h['epoch'] for h in history]
train_loss = [h['train_loss'] for h in history]
val_loss = [h.get('val_loss', 0) for h in history]

plt.plot(epochs, train_loss, label='Training')
plt.plot(epochs, val_loss, label='Validation')
plt.legend()
plt.title('QPMeL Training Progress')
plt.show()
```

## 🚨 Troubleshooting

### Common Issues

**"No space left on device"**
```bash
# Reduce batch size and use fewer triplets
python train_qpmel.py --batch-size 4 --num-triplets 1000
```

**"CUDA out of memory"**
```bash
# Force CPU training
CUDA_VISIBLE_DEVICES="" python train_qpmel.py --batch-size 8
```

**"Quantum circuit depth exceeds limit"**
```bash
# Reduce qubits or layers
python train_qpmel.py --n-qubits 2 --n-layers 1
```

**"Training loss not decreasing"**
```bash
# Try different learning rate and enable QRC
python train_qpmel.py --lr 0.0005 --enable-qrc --margin 0.3
```

### Validation

```python
# Test if your trained model actually works
from quantum_rerank.core.qpmel_integration import load_qpmel_reranker

reranker = load_qpmel_reranker("models/qpmel_trained.pt")

# Simple semantic test
query = "machine learning algorithms"
docs = [
    "neural networks and deep learning models",    # Should rank high
    "supervised and unsupervised learning",       # Should rank high  
    "cooking recipes and food preparation"        # Should rank low
]

results = reranker.rerank(query, docs, method="quantum")
print("Ranking:")
for i, result in enumerate(results):
    print(f"{i+1}. {result['text'][:50]}... (Score: {result['similarity_score']:.3f})")

# If the food recipe ranks last, your model is working!
```

## 🎯 Next Steps

1. **Start small**: Train on synthetic data to verify everything works
2. **Use real data**: Train on NFCorpus or SentenceTransformers for better quality  
3. **Compare results**: Use `evaluate_qpmel.py` to measure improvements
4. **Integrate**: Replace your existing reranker with the trained QPMeL version
5. **Monitor**: Track performance improvements in your application

## 📚 Further Reading

- **QPMeL Paper**: `docs/Papers Quantum Analysis/Quantum Polar Metric Learning_*.txt`
- **Architecture Details**: `quantum_rerank/ml/qpmel_circuits.py`
- **Training Implementation**: `quantum_rerank/training/qpmel_trainer.py`
- **Integration Guide**: `quantum_rerank/core/qpmel_integration.py`

The key insight is that **your quantum circuit parameters should be trained to understand semantic similarity**, not just randomly initialized. QPMeL does exactly that through quantum triplet loss!