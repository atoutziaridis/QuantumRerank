<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Comprehensive FastAPI Documentation for Quantum-Inspired Semantic Reranking in RAG

**Main Takeaway:**
This guide demonstrates how to build a high-performance, enterprise-grade REST API using FastAPI that accepts embeddings and optional user context, computes quantum-inspired similarity scores via Qiskit/PennyLane, reranks top-K candidates, and provides explainability, scalable batch handling, and deployment guidelines.

## 1. Project Setup

### 1.1. Environment \& Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core libraries
pip install fastapi uvicorn
pip install torch transformers sentence-transformers
pip install qiskit pennylane pennylane-qiskit
pip install loguru pydantic
```


### 1.2. Directory Structure

```
quantum_rerank_api/
├── app/
│   ├── main.py
│   ├── models.py
│   ├── quantum.py
│   ├── reranker.py
│   ├── utils.py
│   └── config.py
├── requirements.txt
└── README.md
```


## 2. FastAPI Server Setup

### 2.1. `models.py`: Request \& Response Schemas

```python
from pydantic import BaseModel, conlist
from typing import List, Optional, Dict

class RerankRequest(BaseModel):
    embeddings: List[conlist(float, min_items=1)]  # e.g., [[0.1,0.2,...], ...]
    candidates: List[conlist(float, min_items=1)]
    user_context: Optional[Dict[str, str]] = None
    top_k: Optional[int] = 50

class RerankResponse(BaseModel):
    reranked_indices: List[int]
    scores: List[float]
    explainability: Optional[List[Dict[str, float]]] = None
```


### 2.2. `main.py`: Application \& Endpoints

```python
from fastapi import FastAPI, HTTPException
from app.models import RerankRequest, RerankResponse
from app.reranker import quantum_reranker
from app.utils import configure_logging

app = FastAPI(title="Quantum-Inspired Reranking API")
logger = configure_logging()

@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    try:
        indices, scores, explain = quantum_reranker(
            query_emb=req.embeddings,
            candidate_embs=req.candidates,
            top_k=req.top_k,
            context=req.user_context
        )
        return RerankResponse(
            reranked_indices=indices,
            scores=scores,
            explainability=explain
        )
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail="Internal reranking error")
```


## 3. Core Components

### 3.1. Classical Preprocessing (`utils.py`)

```python
from sentence_transformers import SentenceTransformer
import torch

_model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_texts(texts: list[str]) -> torch.Tensor:
    return _model.encode(texts, convert_to_tensor=True)
```


### 3.2. Quantum-Inspired Similarity (`quantum.py`)

```python
import pennylane as qml
import torch

dev = qml.device("default.qubit", wires=2)

def fidelity_score(v1: torch.Tensor, v2: torch.Tensor) -> float:
    @qml.qnode(dev)
    def circuit(x, y):
        for i, angle in enumerate(x):
            qml.RX(angle, wires=0)
        for i, angle in enumerate(y):
            qml.RX(angle, wires=1)
        return qml.probs(wires=[0,1])
    probs = circuit(v1.tolist(), v2.tolist())
    # fidelity = (sqrt(p00) + sqrt(p11))**2
    return float((probs[0]**0.5 + probs[3]**0.5)**2)

def projection_score(v1: torch.Tensor, v2: torch.Tensor) -> float:
    # Inner product normalized
    return float(torch.cosine_similarity(v1, v2, dim=0))
```


### 3.3. Reranking Logic (`reranker.py`)

```python
import heapq
from typing import List, Tuple, Optional, Dict
from app.quantum import fidelity_score, projection_score

def quantum_reranker(
    query_emb: List[List[float]],
    candidate_embs: List[List[float]],
    top_k: int = 50,
    context: Optional[Dict[str,str]] = None
) -> Tuple[List[int], List[float], Optional[List[Dict[str, float]]]]:
    # Flatten if batch of one
    query = query_emb[0]
    heap: List[Tuple[float,int,Dict[str,float]]] = []
    for idx, cand in enumerate(candidate_embs):
        f_score = fidelity_score(query, cand)
        p_score = projection_score(query, cand)
        combined = 0.5 * f_score + 0.5 * p_score
        explain = {"fidelity": f_score, "projection": p_score}
        heapq.heappush(heap, (-combined, idx, explain))
        if len(heap) > top_k:
            heapq.heappop(heap)
    results = sorted(heap, reverse=True)
    indices = [idx for (_, idx, _) in results]
    scores  = [ -score for (score, _, _) in results]
    explains= [exp for (_, _, exp) in results]
    return indices, scores, explains
```


## 4. Scalability \& Batch Handling

- **Asynchronous Processing**: Uvicorn with multiple workers.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

- **Batching**: Accept lists of query embeddings; process in loops or vectorize with `torch.vmap`.
- **GPU Acceleration**: Offload classical encoding and tensor ops to GPU (`.to('cuda')`).
- **Caching**: Memoize repeated context patterns or candidate sets with Redis.


## 5. Error Handling, Logging \& Explainability

- **Logging**: Using Loguru in `utils.py`

```python
from loguru import logger
def configure_logging():
    logger.add("logs/api.log", rotation="10 MB", level="INFO")
    return logger
```

- **Validation Errors**: Leverage FastAPI’s automatic 422 responses for invalid schemas.
- **Explainability Output**: Return per-candidate fidelity and projection scores alongside final ranking.


## 6. Deployment Tips

### 6.1. Cloud (AWS/GCP/Azure)

- Containerize with Docker:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "2"]
```

- Use managed Kubernetes (EKS/GKE/AKS) with Horizontal Pod Autoscaler.
- Store large embeddings in S3; use AWS Batch for heavy workloads.


### 6.2. On-Premises

- Deploy on machines with NVIDIA GPUs; install CUDA and set `TORCH_CUDA_ARCH_LIST`.
- Use Nginx as a reverse proxy and SSL terminator.
- Monitor with Prometheus + Grafana; expose `/metrics` via Prometheus FastAPI exporter.


## 7. Example Request \& Response

### Request

```http
POST /rerank HTTP/1.1
Content-Type: application/json

{
  "embeddings": [[0.12, 0.34, …, 0.56]],
  "candidates": [[0.11,0.33,…,0.55], …],
  "user_context": {"session_id":"abc123"},
  "top_k": 75
}
```


### Response

```json
{
  "reranked_indices": [5, 2, 0, …],
  "scores": [0.89, 0.87, 0.85, …],
  "explainability": [
    {"fidelity":0.92,"projection":0.86},
    {"fidelity":0.90,"projection":0.84},
    …
  ]
}
```

**This documentation equips you with a ready-to-deploy FastAPI microservice for quantum-inspired semantic reranking, optimized for enterprise-grade RAG systems.**

