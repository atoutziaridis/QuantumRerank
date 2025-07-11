<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Recommendation for Pre-trained Text Embedding Models in a Quantum-Inspired RAG Reranking Pipeline

**Main Recommendation:**
For a quantum-inspired semantic reranking service within a RAG system, **`multi-qa-mpnet-base-dot-v1`** offers the strongest balance of embedding quality, benchmark performance, batch efficiency, and integration ease. Its 768-dimensional dense vectors deliver top-tier MS MARCO and BEIR scores, it is fully PyTorch-compatible, supports high-throughput batches, and can be wrapped in quantum-inspired similarity functions with minimal overhead.

## Comparative Overview of Candidate Models

| Model | Architecture | Embedding Size | MS MARCO MRR@10 | TREC DL19 NDCG@10 | BEIR Retrieval | Throughput (V100 GPU) | Edge Suitability |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| all-MiniLM-L6-v2 [^1] | MiniLM (6-layer) | 384 | ~33 (MRR@100) | ~67 (NDCG@10) | Moderate | ~18 k qps CPU/<br>750 GPU | Excellent (small size) |
| multi-qa-mpnet-base-dot-v1 [^2] [^3] | MPNet-based | 768 | 70.66% | 71.18% | Strong | ~4 k qps GPU<br>170 CPU | Good (mid-range) |
| BAAI/bge-large-en-v1.5 [^4] | BERT-based (335 M) | 1024 | — | — | State-of-art | Embedding: ~54 retrieval@MTEB | Moderate (large) |

- **all-MiniLM-L6-v2**
    - Dimensions: 384
    - Speed: ~18 000 queries/s (GPU); 750 qps (CPU) [^5]
    - Benchmark: NDCG@10 ≈ 67 on TREC DL19; MS MARCO MRR@100 ≈ 33 [^2]
    - Pros: Small footprint, edge-friendly.
    - Cons: Lower retrieval quality than larger models.
- **multi-qa-mpnet-base-dot-v1**
    - Dimensions: 768
    - Speed: ~4 000 qps (GPU); 170 qps (CPU) [^2]
    - Benchmark: MRR@10 70.66%; NDCG@10 71.18% on MS MARCO reranking [^2][^3]
    - Pros: Top retrieval metrics, full Python/PyTorch integration, strong out-of-domain BEIR performance.
    - Cons: Larger embedding size than MiniLM, moderate compute.
- **bge-large-en-v1.5**
    - Dimensions: 1024
    - Benchmarks: Retrieval 54.29 (MTEB); Reranking 60.03 (MTEB) [^4]
    - Pros: State-of-the-art embedding quality on heterogeneous IR tasks, BEIR-ready scripts.
    - Cons: Heaviest compute, largest memory footprint, edge-deployment challenging.


## Suitability for Quantum-Inspired Similarity Metrics

- **Vector Dimensions:**
    - Quantum-inspired fidelity or projection metrics benefit from moderate vector sizes (≥ 384).
    - 768-dim embeddings (MPNet) strike a balance between expressivity and quantum circuit complexity.
- **Batch Processing:**
    - PyTorch’s `DataLoader` can batch-encode 50–100 passages with `multi-qa-mpnet-base-dot-v1` on a single V100 with minimal slowdown, supporting throughputs ≈ 1000 qps in reranking loops.
- **Framework Compatibility:**
    - All three models are loadable via HuggingFace **`SentenceTransformer`** or **`AutoModel`**, enabling seamless wrapping in PennyLane/Qiskit kernels for quantum-inspired similarity computations.


## Integration Guide

### 1. Initial Retrieval with FAISS

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load MPNet embedder
embedder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Encode corpus
corpus = [...]  # list of documents
corpus_emb = embedder.encode(corpus, batch_size=64, convert_to_numpy=True)

# Build FAISS index
d = corpus_emb.shape[^1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(corpus_emb)
index.add(corpus_emb)

# Retrieve top-K candidates
query_emb = embedder.encode("Your query", convert_to_numpy=True)
faiss.normalize_L2(query_emb)
_, candidate_ids = index.search(query_emb, k=100)
candidates = [corpus[i] for i in candidate_ids[^0]]
```


### 2. Quantum-Inspired Reranking Loop

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Prepare quantum-inspired similarity (e.g., fidelity)
def fidelity_similarity(v1, v2):
    # v1, v2: numpy vectors
    return np.dot(v1, v2)  # simplistic; replace with quantum kernel

# Load MPNet for reranking embeddings
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')
model = AutoModel.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1').eval()

def encode_batch(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state[:,0]
    return torch.nn.functional.normalize(outputs, dim=1).cpu().numpy()

# Rerank top-K
query_vec = encode_batch(["Your query"])[^0]
cand_vecs = encode_batch(candidates)
scores = [fidelity_similarity(query_vec, vec) for vec in cand_vecs]
ranked = sorted(zip(candidates, scores), key=lambda x: x[^1], reverse=True)
```


## Final Recommendation

**Select `multi-qa-mpnet-base-dot-v1`** for its superior MS MARCO (MRR@10 70.66%, NDCG@10 71.18%) and robust BEIR performance, combined with manageable embedding size and full compatibility with PyTorch, FAISS, Qiskit, and PennyLane. It balances **retrieval effectiveness**, **batch efficiency**, and **ease of quantum-inspired metric integration**, making it the optimal choice for high-quality dense embeddings in a hybrid quantum-classical RAG reranking service.

<div style="text-align: center">⁂</div>

[^1]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[^2]: https://www.sbert.net/docs/pretrained-models/msmarco-v5.html

[^3]: https://www.promptlayer.com/models/multi-qa-mpnet-base-dot-v1-bf35

[^4]: https://huggingface.co/BAAI/bge-large-en-v1.5

[^5]: https://www.promptlayer.com/models/ms-marco-minilm-l6-v2

[^6]: https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1

[^7]: https://sourceforge.net/projects/bge-large-en-v1-5/

[^8]: https://www.dhiwise.com/post/sentence-embeddings-all-minilm-l6-v2

[^9]: https://bge-model.com/bge/bge_v1_v1.5.html

[^10]: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2

[^11]: https://arxiv.org/pdf/2208.06959.pdf

[^12]: https://www.sbert.net/docs/pretrained-models/ce-msmarco.html

[^13]: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

[^14]: https://dataloop.ai/library/model/baai_bge-large-en-v15/

[^15]: https://www.aimodels.fyi/models/huggingFace/ms-marco-minilm-l-6-v2-cross-encoder

[^16]: https://deepinfra.com/sentence-transformers/multi-qa-mpnet-base-dot-v1

[^17]: https://bge-model.com/API/evaluation/beir.html

[^18]: https://dataloop.ai/library/model/sentence-transformers_multi-qa-mpnet-base-dot-v1/

[^19]: https://www.aimodels.fyi/models/huggingFace/bge-en-icl-baai

[^20]: https://blog.metarank.ai/from-zero-to-semantic-search-embedding-model-592e16d94b61

[^21]: https://huggingface.co/datasets/microsoft/ms_marco

[^22]: https://docs.llamaindex.ai/en/stable/examples/evaluation/BeirEvaluation/

[^23]: https://blog.csdn.net/gitblog_02086/article/details/145204763

[^24]: https://github.com/beir-cellar/beir/issues/129

[^25]: https://www.aimodels.fyi/models/huggingFace/multi-qa-mpnet-base-dot-v1-sentence-transformers

[^26]: https://bge-model.com/tutorial/5_Reranking/5.3.html

[^27]: https://dataloop.ai/library/model/baai_bge-reranker-large/

[^28]: https://zilliz.com/ai-models/bge-reranker-base

[^29]: https://www.atlantis-press.com/article/126004096.pdf

[^30]: https://huggingface.co/model-embeddings/multi-qa-mpnet-base-dot-v1

[^31]: https://discuss.huggingface.co/t/cannot-reproduce-the-baai-bge-reranker-large-re-ranker-model-results/61656

[^32]: https://gist.github.com/szerintedmi/74e3f7b8e22132052df7938c7ad64a4c

[^33]: https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html

[^34]: https://huggingface.co/HgThinker/multi-qa-mpnet-base-dot-v1

